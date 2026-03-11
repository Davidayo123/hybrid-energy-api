from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from model_engine import RealTimeHybridForecaster

app = FastAPI()
# Initialize the engine once
forecaster = RealTimeHybridForecaster()

class GridDataPayload(BaseModel):
    current_features: list
    history_true: list
    history_features: list

@app.post("/predict_load")
async def predict_load(payload: GridDataPayload):
    try:
        # Just pass the lists directly to the engine
        result = forecaster.predict_next_hour(
            payload.current_features, 
            payload.history_true, 
            payload.history_features
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
