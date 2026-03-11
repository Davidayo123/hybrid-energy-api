import numpy as np
import lightgbm as lgb
from tensorflow.keras.models import load_model
import joblib

class RealTimeHybridForecaster:
    def __init__(self, gru_model_path='tegru.keras', lgb_model_path='lightgbm.pkl'):
        print("⚙️ Booting up the Hybrid Forecaster Engine...")
        
        # 1. Load the Pre-Trained Base Models
        self.te_gru = load_model(gru_model_path)
        self.lgb_model = joblib.load(lgb_model_path)
        
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
        
        print("✅ Models loaded and 7D parameters locked.")

    def run_mh_control_loop(self, hist_true, hist_gru, hist_lgb):
        """
        Runs the 28-step Markov Chain to dynamically adjust weight and bias
        based on the grid's volatility over the last 120 hours.
        """
        for _ in range(self.mh_steps):
            # Propose new 2D states
            proposed_w = np.clip(self.current_w + np.random.normal(0, self.step_w), 0.0, 1.0)
            proposed_b = self.current_b + np.random.normal(0, self.step_b)
            
            # Reconstruct history with current vs. proposed states
            pred_current = (self.current_w * hist_gru) + ((1 - self.current_w) * hist_lgb) + self.current_b
            pred_proposed = (proposed_w * hist_gru) + ((1 - proposed_w) * hist_lgb) + proposed_b
            
            # Recency-weighted error calculation
            error_current = np.sum(self.exp_weights * (hist_true - pred_current)**2)
            error_proposed = np.sum(self.exp_weights * (hist_true - pred_proposed)**2)
            
            delta_error = error_proposed - error_current
            
            # Accept or reject the new thermodynamic state
            if delta_error < 0 or np.random.rand() < np.exp(-delta_error / self.temperature):
                self.current_w = proposed_w
                self.current_b = proposed_b

    def predict_next_hour(self, current_features, history_true, history_features):
        """
        The main function Group 3's API will call. 
        It formats data independently for the GRU and LightGBM models.
        """
        # 1. Format data correctly for BOTH models
        # GRU strictly needs 3D data: (Batch, Time Steps, Features)
        gru_hist_feat = history_features.reshape(self.window_size, 1, -1)
        gru_curr_feat = current_features.reshape(1, 1, -1)
        
        # LightGBM strictly needs 2D data: (Batch, Features)
        lgb_hist_feat = history_features.reshape(self.window_size, -1)
        lgb_curr_feat = current_features.reshape(1, -1)
        
        # 2. Generate historical predictions for the 120-hour memory buffer
        hist_gru = self.te_gru.predict(gru_hist_feat, verbose=0).flatten()
        hist_lgb = self.lgb_model.predict(lgb_hist_feat)
        
        # 3. Run the dynamic control loop to find the perfect current_w and current_b
        self.run_mh_control_loop(history_true, hist_gru, hist_lgb)
        
        # 4. Generate the pure predictions for the single NEXT hour
        next_gru = self.te_gru.predict(gru_curr_feat, verbose=0).flatten()[0]
        next_lgb = self.lgb_model.predict(lgb_curr_feat)[0]
        
        # 5. Apply the optimized state for the final output
        final_prediction = (self.current_w * next_gru) + ((1 - self.current_w) * next_lgb) + self.current_b
        
        return {
            "predicted_load_kw": float(final_prediction),
            "diagnostics": {
                "active_gru_weight": float(self.current_w),
                "active_bias_correction": float(self.current_b)
            }
        }

        }
