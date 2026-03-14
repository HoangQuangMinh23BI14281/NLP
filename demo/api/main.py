import os
import re
from typing import Any, Dict, List, Optional

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
    skill_coverage: List[Dict[str, Any]]

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

        def normalize_text(value: str) -> str:
            cleaned = re.sub(r"[^a-z0-9\s+]", " ", value.lower())
            return " ".join(cleaned.split())

        def dedupe(values: List[str]) -> List[str]:
            seen = set()
            output: List[str] = []
            for value in values:
                if value and value not in seen:
                    seen.add(value)
                    output.append(value)
            return output

        def phrase_match(candidate: str, required: str) -> bool:
            candidate_norm = normalize_text(candidate)
            required_norm = normalize_text(required)
            if not candidate_norm or not required_norm:
                return False

            if candidate_norm == required_norm:
                return True

            candidate_tokens = set(candidate_norm.split())
            required_tokens = set(required_norm.split())
            if not candidate_tokens or not required_tokens:
                return False

            # For single-token requirements, avoid fuzzy partial matches.
            if len(required_tokens) == 1:
                return list(required_tokens)[0] in candidate_tokens

            # Prefer phrase containment for multi-token requirements.
            if required_norm in candidate_norm or candidate_norm in required_norm:
                return True

            overlap = len(candidate_tokens & required_tokens) / len(required_tokens)
            return overlap >= 0.8

        def binary_score(required_values: List[str], candidate_values: List[str]) -> Optional[float]:
            if not required_values:
                return None
            matched = any(
                phrase_match(candidate, required)
                for required in required_values
                for candidate in candidate_values
            )
            return 100.0 if matched else 0.0

        candidate_skills_filtered = dedupe(
            [
                normalize_text(e["text"])
                for e in entities
                if e.get("type") == "SKILL" and float(e.get("score", 0.0)) >= 0.45
            ]
        )
        # Fallback for models that do not produce useful confidence values.
        candidate_skills_raw = dedupe([normalize_text(e["text"]) for e in entities if e.get("type") == "SKILL"])
        candidate_skills = candidate_skills_filtered if candidate_skills_filtered else candidate_skills_raw

        candidate_roles = dedupe([normalize_text(e["text"]) for e in entities if e.get("type") == "ROLE"])
        candidate_locs = dedupe([normalize_text(e["text"]) for e in entities if e.get("type") == "LOC"])
        candidate_exps = dedupe([normalize_text(e["text"]) for e in entities if e.get("type") == "EXP"])

        matches: List[Dict[str, Any]] = []
        skill_coverage: List[Dict[str, Any]] = []

        for skill in candidate_skills:
            matched_companies = []
            for company in COMPANIES:
                req_skills = dedupe([normalize_text(s) for s in company["requirements"].get("SKILL", [])])
                if any(phrase_match(skill, req_skill) for req_skill in req_skills):
                    matched_companies.append(company["name"])

            skill_coverage.append(
                {
                    "skill": skill,
                    "matched_companies": matched_companies,
                    "match_count": len(matched_companies),
                    "total_companies": len(COMPANIES),
                }
            )

        skill_coverage.sort(key=lambda item: (-item["match_count"], item["skill"]))

        for company in COMPANIES:
            req_skills = dedupe([normalize_text(s) for s in company["requirements"].get("SKILL", [])])
            req_roles = dedupe([normalize_text(r) for r in company["requirements"].get("ROLE", [])])
            req_locs = dedupe([normalize_text(l) for l in company["requirements"].get("LOC", [])])
            req_exps = dedupe([normalize_text(x) for x in company["requirements"].get("EXP", [])])

            matched_required_skills = [
                req_skill for req_skill in req_skills
                if any(phrase_match(candidate_skill, req_skill) for candidate_skill in candidate_skills)
            ]

            matched_candidate_skills = [
                candidate_skill for candidate_skill in candidate_skills
                if any(phrase_match(candidate_skill, req_skill) for req_skill in req_skills)
            ]

            skill_score = None
            if req_skills:
                skill_recall = len(matched_required_skills) / len(req_skills)
                skill_precision = len(set(matched_candidate_skills)) / len(candidate_skills) if candidate_skills else 0.0
                if skill_precision + skill_recall > 0:
                    skill_f1 = (2 * skill_precision * skill_recall) / (skill_precision + skill_recall)
                else:
                    skill_f1 = 0.0
                skill_score = skill_f1 * 100

            role_score = binary_score(req_roles, candidate_roles)
            loc_score = binary_score(req_locs, candidate_locs)
            exp_score = binary_score(req_exps, candidate_exps)

            weighted_parts = []
            if skill_score is not None:
                weighted_parts.append((skill_score, 0.6))
            if role_score is not None:
                weighted_parts.append((role_score, 0.25))
            if loc_score is not None:
                weighted_parts.append((loc_score, 0.1))
            if exp_score is not None:
                weighted_parts.append((exp_score, 0.05))

            if weighted_parts:
                total_weight = sum(weight for _, weight in weighted_parts)
                overall_score = sum(score * weight for score, weight in weighted_parts) / total_weight
            else:
                overall_score = 0.0

            matches.append(
                {
                    "company_id": company["id"],
                    "company_name": company["name"],
                    "description": company["description"],
                    "match_score": round(overall_score, 2),
                    "matched_skills": sorted(set(matched_required_skills)),
                    "missing_skills": [
                        s for s in req_skills if s not in matched_required_skills
                    ],
                    "matched_skill_count": len(set(matched_required_skills)),
                    "required_skill_count": len(req_skills),
                }
            )

        matches.sort(key=lambda item: item["match_score"], reverse=True)
        return {"candidate_entities": entities, "matches": matches, "skill_coverage": skill_coverage}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
