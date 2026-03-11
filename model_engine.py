import os
import numpy as np
import lightgbm as lgb
from tensorflow.keras.models import load_model
import joblib

class RealTimeHybridForecaster:
    def __init__(self, gru_model_path='te_gru_custom.keras', lgb_model_path='lightgbm_baseline.pkl'):
        print("⚙️ Booting up the Hybrid Forecaster Engine...")
        
        # Determine the absolute path to the files (Fixes Render pathing issues)
        base_path = os.path.dirname(os.path.abspath(__file__))
        gru_full_path = os.path.join(base_path, gru_model_path)
        lgb_full_path = os.path.join(base_path, lgb_model_path)
        
        # 1. Load the Pre-Trained Base Models
        try:
            self.te_gru = load_model(gru_full_path)
            self.lgb_model = joblib.load(lgb_full_path)
            print("✅ Models loaded and 7D parameters locked.")
        except Exception as e:
            print(f"❌ CRITICAL LOAD ERROR: {str(e)}")
            raise e
        
        # 2. Hardcode the Champion 7D "God-Mode" Parameters
        self.window_size = 120
        self.mh_steps = 28
        self.step_w = 0.00129
        self.step_b = 0.01133
        self.temperature = 1.66e-06
        self.decay_rate = 0.00443
        
        # 3. Initialize the Thermodynamic States
        self.current_w = 0.16034
        self.current_b = 0.0
        
        # Pre-calculate the exponential memory weights for the 120-hour window
        self.exp_weights = np.exp(self.decay_rate * np.arange(self.window_size))
        self.exp_weights /= np.sum(self.exp_weights)

    def run_mh_control_loop(self, hist_true, hist_gru, hist_lgb):
        """
        Calculates weights based on historical volatility.
        """
        # Ensure all are numpy arrays for the math
        hist_true = np.array(hist_true)
        hist_gru = np.array(hist_gru)
        hist_lgb = np.array(hist_lgb)

        for _ in range(self.mh_steps):
            proposed_w = np.clip(self.current_w + np.random.normal(0, self.step_w), 0.0, 1.0)
            proposed_b = self.current_b + np.random.normal(0, self.step_b)
            
            pred_current = (self.current_w * hist_gru) + ((1 - self.current_w) * hist_lgb) + self.current_b
            pred_proposed = (proposed_w * hist_gru) + ((1 - proposed_w) * hist_lgb) + proposed_b
            
            error_current = np.sum(self.exp_weights * (hist_true - pred_current)**2)
            error_proposed = np.sum(self.exp_weights * (hist_true - pred_proposed)**2)
            
            delta_error = error_proposed - error_current
            
            if delta_error < 0 or np.random.rand() < np.exp(-delta_error / self.temperature):
                self.current_w = proposed_w
                self.current_b = proposed_b

    def predict_next_hour(self, current_features, history_true, history_features):
        # Ensure inputs are numpy
        curr_feat_np = np.array(current_features)
        hist_feat_np = np.array(history_features)

        # 1. Format for GRU (Sequence Mode)
        # Your model wants (1, 120, 17)
        gru_input_seq = hist_feat_np.reshape(1, self.window_size, -1)
        
        # 2. Predict next hour from both models
        # GRU uses the 120-hour sequence to predict the next point
        next_gru = self.te_gru.predict(gru_input_seq, verbose=0).flatten()[0]
        
        # LightGBM just uses the 1 most recent hour
        lgb_input_curr = curr_feat_np.reshape(1, -1)
        next_lgb = self.lgb_model.predict(lgb_input_curr)[0]
        
        # 3. Simulate history for the MH Loop
        # We broadcast the GRU prediction to keep the loop logic stable
        hist_gru_sim = np.full(self.window_size, next_gru)
        hist_lgb_sim = self.lgb_model.predict(hist_feat_np.reshape(self.window_size, -1))
        
        # 4. Run the Control Loop
        self.run_mh_control_loop(history_true, hist_gru_sim, hist_lgb_sim)
        
        # 5. Final Output
        final_prediction = (self.current_w * next_gru) + ((1 - self.current_w) * next_lgb) + self.current_b
        
        return {
            "predicted_load_kw": float(final_prediction),
            "diagnostics": {
                "active_gru_weight": float(self.current_w),
                "active_bias_correction": float(self.current_b)
            }
        }
