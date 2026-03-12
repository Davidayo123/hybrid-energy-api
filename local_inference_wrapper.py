import numpy as np
import joblib
import tflite_runtime.interpreter as tflite  # Lightweight library for the Pi 5

class LocalEdgeForecaster:
    def __init__(self, tflite_path='te_gru_quantized.tflite', lgb_path='lightgbm_baseline.pkl', scaler_path='scaler.joblib'):
        print("⚙️ Booting Master-Slave AI Hub...")
        
        # 1. Load Preprocessing & LightGBM
        self.scaler = joblib.load(scaler_path)
        self.lgb_model = joblib.load(lgb_path)
        
        # 2. Load TFLite Model (TE-GRU)
        self.interpreter = tflite.Interpreter(model_path=tflite_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Metropolis-Hastings Adaptive Weights (Hybrid Logic)
        self.current_w = 0.5
        self.current_b = 0.0
        
        # Uncertainty Estimation (Upper Bound Margin)
        # Adjust this value based on your model's actual RMSE/MAE
        self.uncertainty_margin_kw = 0.15 
        
        print("✅ Hybrid Edge Models Loaded Successfully!")

    def predict(self, current_hour_data):
        """
        Takes raw ESP32 sensor array (1, 17) and outputs Smart Grid thresholds.
        ⚠️ Format: [Energy, Temp, Humidity, Lux, Occupancy, Hour, DayOfWeek...]
        """
        scaled_curr = np.copy(current_hour_data)
        
        # Scale ONLY the 4 physical continuous variables
        scaled_curr[:, 0:4] = self.scaler.transform(scaled_curr[:, 0:4])
        
        # LightGBM Inference
        pred_lgb = float(self.lgb_model.predict(scaled_curr)[0])
        
        # TE-GRU Inference
        input_data = np.array(scaled_curr, dtype=np.float32)
        input_data = input_data.reshape(self.input_details[0]['shape']) 
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke() 
        pred_gru = float(self.interpreter.get_tensor(self.output_details[0]['index'])[0][0])
        
        # Calculate Mean Prediction
        mean_kw = (self.current_w * pred_gru) + ((1 - self.current_w) * pred_lgb) + self.current_b
        
        # Calculate Upper Bound for Group 3's Load Shedding Logic
        upper_bound_kw = mean_kw + self.uncertainty_margin_kw
        
        # Return a dictionary so the Django/MQTT backend can easily parse it
        return {
            "mean_prediction_kw": round(mean_kw, 3),
            "upper_bound_kw": round(upper_bound_kw, 3)
        }

# ==========================================
# GROUP 3 INTEGRATION TEMPLATE
# ==========================================
if __name__ == "__main__":
    # 1. AI initialized on Pi boot
    forecaster = LocalEdgeForecaster()
    
    # 2. Simulated ESP32 Data arriving via Mosquitto MQTT
    # This array represents the 17 features queried from the local SQLite database
    live_sensor_array = np.random.rand(1, 17) 
    
    # 3. Run Inference
    ai_results = forecaster.predict(live_sensor_array)
    print(f"📊 AI Mean Forecast: {ai_results['mean_prediction_kw']} kW")
    print(f"🚨 AI Upper Bound (Safety Limit): {ai_results['upper_bound_kw']} kW")
    
    # 4. Simulated Group 3 Control Logic (Relay Actuation)
    simulated_battery_soc = 65.0 # Voltage divider reading
    battery_stable = True        # 3-Time Battery Lag Algorithm result
    
    if battery_stable:
        if ai_results['upper_bound_kw'] > 2.0 and simulated_battery_soc < 50.0:
            print("⚡ Action: Switching to Smart Mode C (Conservation). Shedding Class A & B SSRs.")
        elif ai_results['upper_bound_kw'] > 1.0 and simulated_battery_soc < 80.0:
            print("⚡ Action: Switching to Smart Mode B (Average). Shedding Class A SSRs.")
        else:
            print("⚡ Action: Switching to Smart Mode A (Maximum). All Relays Authorized.")