import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import numpy as np

# Import the "Brain" we just built
from model_engine import RealTimeHybridForecaster

print("🚀 Initializing FastAPI Server...")

# 1. Boot up the API and the AI Engine
app = FastAPI(title="Hybrid Energy Forecaster API", version="1.0")

# Ensure we use the exact filenames on GitHub
forecaster = RealTimeHybridForecaster(
    gru_model_path='te_gru_custom.keras', 
    lgb_model_path='lightgbm_baseline.pkl'
)

# 2. Define the Data Contract (What Group 3 MUST send you)
class GridDataPayload(BaseModel):
    current_features: list    # 1D array of 17 numbers
    history_true: list       # 1D array of 120 numbers
    history_features: list    # 2D array of (120, 17)

# 3. Create the Endpoint
@app.post("/predict_load")
async def predict_load(payload: GridDataPayload):
    try:
        # Convert lists to NumPy arrays for validation
        curr_feat = np.array(payload.current_features)
        hist_true = np.array(payload.history_true)
        hist_feat = np.array(payload.history_features)
        
        # Check if Group 3 sent the correct 120-hour memory buffer
        if len(hist_true) != forecaster.window_size:
            raise ValueError(f"history_true must be exactly {forecaster.window_size} hours long.")

        # Ask the AI Engine for the prediction
        # We pass raw arrays; model_engine.py handles the 3D/2D reshaping
        prediction_result = forecaster.predict_next_hour(
            current_features=curr_feat,
            history_true=hist_true,
            history_features=hist_feat
        )
        
        return prediction_result
        
    except Exception as e:
        # Send detailed error back to the client
        raise HTTPException(status_code=400, detail=str(e))

# 4. Entry point for local testing (Render uses its own start command)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
