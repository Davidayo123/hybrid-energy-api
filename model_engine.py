import os
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model

print(f"⚙️ TENSORFLOW VERSION DETECTED: {tf.__version__}")

class RealTimeHybridForecaster:
    def __init__(self, gru_model_path='te_gru_custom.keras', lgb_model_path='lightgbm_baseline.pkl'):
        self.window_size = 120
        self.current_w = 0.5
        self.current_b = 0.0
        
        base_path = os.path.dirname(os.path.abspath(__file__))
        gru_full_path = os.path.join(base_path, gru_model_path)
        lgb_full_path = os.path.join(base_path, lgb_model_path)
        
        print(f"🔥 ATTEMPTING TO LOAD GRU FROM: {gru_full_path}")
        
        # NO SAFETY NET. IF THIS FAILS, THE SERVER DIES AND LOGS THE TRUTH.
        self.te_gru = load_model(gru_full_path)
        
        print(f"🔥 ATTEMPTING TO LOAD LGB FROM: {lgb_full_path}")
        self.lgb_model = joblib.load(lgb_full_path)
        
        print("✅ MODELS LOADED SUCCESSFULLY")

    def predict_next_hour(self, current_features, history_true, history_features):
        curr_feat = np.array(current_features).reshape(1, -1)
        hist_feat = np.array(history_features).reshape(1, 120, -1)
        
        pred_gru = float(self.te_gru.predict(hist_feat, verbose=0).flatten()[0])
        pred_lgb = float(self.lgb_model.predict(curr_feat)[0])
        
        final = (self.current_w * pred_gru) + ((1 - self.current_w) * pred_lgb) + self.current_b
        return {"predicted_load_kw": final, "status": "live"}
