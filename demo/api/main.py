import os
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformers import AutoTokenizer
from models import load_ner_model
from utils import NERInference
from companies import COMPANIES

app = FastAPI(
    title="NER IT Recruitment API",
    description="API for Named Entity Recognition in IT Job Descriptions",
    version="1.0.0"
)

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
# Model resolution:
# 1) DEMO_MODEL_PATH env var (if set)
# 2) New default CRF model folder
# 3) Legacy fallback folder names
def resolve_model_path() -> str:
    env_model_path = os.getenv("DEMO_MODEL_PATH")
    if env_model_path:
        return env_model_path

    candidates = [
        os.path.join(BASE_DIR, "backend", "bert-base-cased_CRF"),
        os.path.join(BASE_DIR, "backend", "debug_model_CRF_download"),
        os.path.join(BASE_DIR, "backend", "debug_model_Base_CE_download"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    # Keep first candidate for clear error/warning message if nothing exists.
    return candidates[0]


MODEL_PATH = resolve_model_path()
ARCH_TYPE = os.getenv("DEMO_ARCH_TYPE") or ("CRF" if "CRF" in os.path.basename(MODEL_PATH) else "Base_CE")

if not os.path.exists(MODEL_PATH):
    print(f"Warning: Model path {MODEL_PATH} not found. Ensure models are downloaded.")

# Global model instance
model = None
tokenizer = None
inference_engine = None

EXPECTED_NER_LABELS = ["ROLE", "SKILL", "LOC", "EXP", "SALARY"]


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    tokens: List[str]
    tags: List[str]
    token_scores: List[float]
    entities: List[Dict[str, Any]]


class MatchResponse(BaseModel):
    candidate_entities: List[Dict[str, Any]]
    matches: List[Dict[str, Any]]

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
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "NER API is running. Go to /docs for Swagger UI."}


@app.get("/health")
async def health() -> Dict[str, bool]:
    return {"ok": True, "model_loaded": inference_engine is not None}


@app.get("/labels")
async def labels() -> Dict[str, List[str]]:
    # Exposes expected NER fields so the frontend can display missing guidance consistently.
    return {"required_fields": EXPECTED_NER_LABELS}

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    if not inference_engine:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        result = inference_engine.predict(request.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/match", response_model=MatchResponse)
async def match(request: PredictRequest):
    if not inference_engine:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        result = inference_engine.predict(request.text)
        entities = result.get("entities", [])

        candidate_skills = [e["text"].lower().strip() for e in entities if e.get("type") == "SKILL"]
        candidate_roles = [e["text"].lower().strip() for e in entities if e.get("type") == "ROLE"]
        candidate_locs = [e["text"].lower().strip() for e in entities if e.get("type") == "LOC"]
        candidate_exps = [e["text"].lower().strip() for e in entities if e.get("type") == "EXP"]

        matches: List[Dict[str, Any]] = []

        for company in COMPANIES:
            req_skills = [s.lower() for s in company["requirements"].get("SKILL", [])]
            req_roles = [r.lower() for r in company["requirements"].get("ROLE", [])]
            req_locs = [l.lower() for l in company["requirements"].get("LOC", [])]
            req_exps = [x.lower() for x in company["requirements"].get("EXP", [])]

            matched_skills = [s for s in candidate_skills if any(rs in s or s in rs for rs in req_skills)]
            skill_score = (len(matched_skills) / len(req_skills)) * 100 if req_skills else 100

            role_match = any(any(rr in cr or cr in rr for rr in req_roles) for cr in candidate_roles)
            role_score = 100 if role_match else 0

            loc_match = any(any(rl in cl or cl in rl for rl in req_locs) for cl in candidate_locs)
            loc_score = 100 if loc_match else 0
            if not req_locs:
                loc_score = 100

            exp_match = any(any(rx in cx or cx in rx for rx in req_exps) for cx in candidate_exps)
            exp_score = 100 if exp_match else 0
            if not req_exps:
                exp_score = 100

            overall_score = (skill_score * 0.5) + (role_score * 0.3) + (loc_score * 0.1) + (exp_score * 0.1)

            matches.append(
                {
                    "company_id": company["id"],
                    "company_name": company["name"],
                    "description": company["description"],
                    "match_score": round(overall_score, 2),
                    "matched_skills": sorted(set(matched_skills)),
                    "missing_skills": [
                        s for s in req_skills if not any(s in ms or ms in s for ms in matched_skills)
                    ],
                }
            )

        matches.sort(key=lambda item: item["match_score"], reverse=True)
        return {"candidate_entities": entities, "matches": matches}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
