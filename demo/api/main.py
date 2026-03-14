import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer
from models import load_ner_model
from utils import NERInference

app = FastAPI(
    title="NER IT Recruitment API",
    description="API for Named Entity Recognition in IT Job Descriptions",
    version="1.0.0"
)

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Default to CRF model as requested
MODEL_PATH = os.path.join(BASE_DIR, "backend", "debug_model_CRF_download")
# OR if you want to switch to Base_CE:
# MODEL_PATH = os.path.join(BASE_DIR, "backend", "debug_model_Base_CE_download")

ARCH_TYPE = "CRF" # or "Base_CE"

if not os.path.exists(MODEL_PATH):
    print(f"Warning: Model path {MODEL_PATH} not found. Ensure models are downloaded.")

# Global model instance
model = None
tokenizer = None
inference_engine = None

class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    tokens: list
    tags: list
    entities: list

@app.on_event("startup")
async def startup_event():
    global model, tokenizer, inference_engine
    try:
        print(f"Loading model from {MODEL_PATH}...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = load_ner_model(MODEL_PATH, arch_type=ARCH_TYPE)
        
        # Get id2label from config
        id2label = model.config.id2label
        inference_engine = NERInference(model, tokenizer, id2label)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")

@app.get("/")
async def root():
    return {"message": "NER API is running. Go to /docs for Swagger UI."}

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    if not inference_engine:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        result = inference_engine.predict(request.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
