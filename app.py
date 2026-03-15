from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from local_inference_wrapper import LocalEdgeForecaster

# 1. Initialize FastAPI
app = FastAPI(title="Smart Grid AI Edge API")

print("⏳ Starting Local FastAPI Server and loading AI models...")
ai_brain = LocalEdgeForecaster()
print("✅ AI API is live and listening on port 5000!")

# 2. Pydantic Model for strict data validation
class SensorData(BaseModel):
    timestamp: str
    temperature_c: float
    humidity: float
    lux: float
    occupancy: int
    lag_1h: float
    lag_2h: float
    lag_3h: float
    lag_24h: float

# 3. The API Endpoint
@app.post("/predict")
def predict_energy(data: SensorData):
    try:
        # Convert the validated Pydantic object into a standard Python dictionary
        incoming_data = data.model_dump() 
        
        # Pass to the Black Box
        results = ai_brain.build_features_and_predict(incoming_data)
        return results
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# 4. Run the server locally using Uvicorn
if __name__ == "__main__":
    # Host 127.0.0.1 keeps it strictly offline on the Raspberry Pi
    uvicorn.run(app, host="127.0.0.1", port=5000)
