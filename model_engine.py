import os
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model

class RealTimeHybridForecaster:
    def __init__(self, gru_model_path='te_gru_custom.keras', lgb_model_path='lightgbm_baseline.pkl'):
        # 1. SET PARAMETERS FIRST (So they always exist no matter what)
        self.window_size = 120
        self.mh_steps = 28
        self.step_w = 0.00129
        self.step_b = 0.01133
        self.temperature = 1.66e-06
        self.decay_rate = 0.00443
        self.current_w = 0.16034
        self.current_b = 0.0
        
        # Pre-calculate exponential weights
        self.exp_weights = np.exp(self.decay_rate * np.arange(self.window_size))
        self.exp_weights /= np.sum(self.exp_weights)

        # 2. FIND AND LOAD MODELS
        base_path = os.path.dirname(os.path.abspath(__file__))
        gru_full_path = os.path.join(base_path, gru_model_path)
        lgb_full_path = os.path.join(base_path, lgb_model_path)

        try:
            print(f"DEBUG: Loading GRU from {gru_full_path}")
            self.te_gru = load_model(gru_full_path)
            print(f"DEBUG: Loading LGB from {lgb_full_path}")
            self.lgb_model = joblib.load(lgb_full_path)
            self.models_loaded = True
            print("✅ SUCCESS: All models active.")
        except Exception as e:
            print(f"⚠️ WARNING: Model load failed ({str(e)}).")
            self.te_gru = None
            self.lgb_model = None
            self.models_loaded = False

    def run_mh_control_loop(self, hist_true, hist_gru, hist_lgb):
        hist_true, hist_gru, hist_lgb = np.array(hist_true), np.array(hist_gru), np.array(hist_lgb)
        for _ in range(self.mh_steps):
            proposed_w = np.clip(self.current_w + np.random.normal(0, self.step_w), 0.0, 1.0)
            proposed_b = self.current_b + np.random.normal(0, self.step_b)
            pred_curr = (self.current_w * hist_gru) + ((1 - self.current_w) * hist_lgb) + self.current_b
            pred_prop = (proposed_w * hist_gru) + ((1 - proposed_w) * hist_lgb) + proposed_b
            err_curr = np.sum(self.exp_weights * (hist_true - pred_curr)**2)
            err_prop = np.sum(self.exp_weights * (hist_true - pred_prop)**2)
            if err_prop < err_curr or np.random.rand() < np.exp(-(err_prop - err_curr) / self.temperature):
                self.current_w, self.current_b = proposed_w, proposed_b

    def predict_next_hour(self, current_features, history_true, history_features):
        # Fallback if models are still booting
        if not self.models_loaded:
            return {"predicted_load_kw": 0.0, "status": "initializing"}

        hist_feat_np = np.array(history_features)
        curr_feat_np = np.array(current_features).reshape(1, -1)
        
        # Prepare 3D for GRU (1, 120, 17)
        gru_input = hist_feat_np.reshape(1, self.window_size, -1)
        next_gru = self.te_gru.predict(gru_input, verbose=0).flatten()[0]
        
        # Prepare 2D for LGB
        next_lgb = self.lgb_model.predict(curr_feat_np)[0]
        
        # Historical simulation for MH loop
        hist_lgb = self.lgb_model.predict(hist_feat_np.reshape(self.window_size, -1))
        hist_gru = np.full(self.window_size, next_gru) 
        
        self.run_mh_control_loop(history_true, hist_gru, hist_lgb)
        
        final_pred = (self.current_w * next_gru) + ((1 - self.current_w) * next_lgb) + self.current_b
        
        return {
            "predicted_load_kw": float(final_pred),
            "diagnostics": {
                "active_gru_weight": float(self.current_w),
                "active_bias_correction": float(self.current_b)
            }
        }
