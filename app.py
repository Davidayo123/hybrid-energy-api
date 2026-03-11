from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import numpy as np

# Import the "Brain" we just built
from model_engine import RealTimeHybridForecaster

print("🚀 Initializing FastAPI Server...")

# 1. Boot up the API and the AI Engine
app = FastAPI(title="Hybrid Energy Forecaster API", version="1.0")
forecaster = RealTimeHybridForecaster(gru_model_path='tegru.keras', lgb_model_path='lightgbm.pkl')

# 2. Define the Data Contract (What Group 3 MUST send you)
class GridDataPayload(BaseModel):
    current_features: list       # The weather/time data for the NEXT hour (1D array)
    history_true: list           # The actual KW load for the LAST 120 hours (1D array)
    history_features: list       # The weather/time data for the LAST 120 hours (2D array)

# 3. Create the Endpoint (The "URL" Group 3 will hit)
@app.post("/predict_load")
async def predict_load(payload: GridDataPayload):
    try:
        # Convert Group 3's JSON lists into NumPy arrays for your AI
        curr_feat_arr = np.array(payload.current_features).reshape(1, 1, -1) # Assuming GRU needs 3D shape (1, 1, features)
        hist_true_arr = np.array(payload.history_true)
        hist_feat_arr = np.array(payload.history_features)
        
        # Check if Group 3 sent the correct 120-hour memory buffer
        if len(hist_true_arr) != forecaster.window_size:
            raise ValueError(f"history_true must be exactly {forecaster.window_size} hours long.")

        # Ask the AI Engine for the prediction
        prediction_result = forecaster.predict_next_hour(
            current_features=curr_feat_arr,
            history_true=hist_true_arr,
            history_features=hist_feat_arr
        )
        
        # Send the KW prediction back to Group 3
        return prediction_result
        
    except Exception as e:
        # If Group 3 sends bad data, send them an error message
        raise HTTPException(status_code=400, detail=str(e))

# 4. Run the Server
if __name__ == "__main__":
    print("⚡ Server is live! Waiting for Group 3's controller...")
    uvicorn.run(app, host="0.0.0.0", port=8000)