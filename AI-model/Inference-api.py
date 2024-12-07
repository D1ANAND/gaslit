from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
from typing import List, Dict
import torch
import uvicorn

app = FastAPI(
    title="Inference-Engine API",
    description="API for running inference on the Inference-Engine model",
    version="1.0.0"
)

# Pydantic models for request/response validation
class InferenceRequest(BaseModel):
    text: str
    parameters: Dict = {}

class InferenceResponse(BaseModel):
    result: List[Dict]
    model_name: str
    processing_time: float

# Global variables
MODEL_NAME = "Inference-Engine"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize the model
@app.on_event("startup")
async def load_model():
    global model
    try:
        model = pipeline(
            "text-generation",
            model=MODEL_NAME,
            device=device
        )
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise RuntimeError("Failed to load model")

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_name": MODEL_NAME,
        "device": device
    }

# Main inference endpoint
@app.post("/predict", response_model=InferenceResponse)
async def predict(request: InferenceRequest):
    try:
        import time
        start_time = time.time()

        # Merge default parameters with user parameters
        default_params = {
            "max_length": 100,
            "num_return_sequences": 1,
            "temperature": 0.7,
            "top_p": 0.9,
        }
        params = {**default_params, **request.parameters}

        # Run inference
        outputs = model(
            request.text,
            **params
        )

        # Calculate processing time
        processing_time = time.time() - start_time

        return InferenceResponse(
            result=outputs,
            model_name=MODEL_NAME,
            processing_time=processing_time
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {str(e)}"
        )

# Model information endpoint
@app.get("/model-info")
async def model_info():
    return {
        "model_name": MODEL_NAME,
        "device": device,
        "default_parameters": {
            "max_length": 100,
            "num_return_sequences": 1,
            "temperature": 0.7,
            "top_p": 0.9
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )