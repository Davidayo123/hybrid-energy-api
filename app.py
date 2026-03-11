from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model_engine import RealTimeHybridForecaster

app = FastAPI()
# Initialize the engine once with the EXACT filenames
forecaster = RealTimeHybridForecaster(gru_model_path='tegru.keras', lgb_model_path='lightgbm.pkl')

class GridDataPayload(BaseModel):
    current_features: list
    history_true: list
    history_features: list

@app.post("/predict_load")
async def predict_load(payload: GridDataPayload):
    try:
        result = forecaster.predict_next_hour(
            payload.current_features, 
            payload.history_true, 
            payload.history_features
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
        
