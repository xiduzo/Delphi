import os
import re
import sys
from typing import List, Optional, Dict, Any

import torch
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Ensure repo root is importable
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from model import Delphi, DelphiConfig  # noqa: E402


class PatientEvent(BaseModel):
    """A single patient event with an ICD10 code and age."""
    code: str = Field(..., description="ICD10 code, e.g. 'I10'", example="I10")
    age_at_event: float = Field(..., description="Age in years at the event", example=45.5)


class PredictRequest(BaseModel):
    """Request body for prediction endpoint."""
    patient: List[PatientEvent] = Field(..., description="List of patient events (ICD10 codes with ages)")
    timeframe_years: Optional[float] = Field(
        None,
        description="Timeframe in years for predictions (e.g., 5.0 for next 5 years). If not provided, predictions are for the immediate next event.",
        ge=0.0,
        example=5.0
    )


class RankedPrediction(BaseModel):
    """A single ranked prediction result."""
    index: int = Field(..., description="Index of the prediction in the model output")
    code: Optional[str] = Field(None, description="ICD10 code if available")
    label: Optional[str] = Field(None, description="Human-readable label for the condition")
    probability: float = Field(..., description="Predicted probability (0.0 to 1.0)", ge=0.0, le=1.0)


class PredictResponse(BaseModel):
    """Response from the prediction endpoint."""
    ranked: List[RankedPrediction] = Field(..., description="Top 10 ranked predictions")
    used_patient: List[PatientEvent] = Field(..., description="Patient events that were actually used (after filtering)")
    warnings: List[str] = Field(default_factory=list, description="Any warnings about the input data")


class CodeInfo(BaseModel):
    """Information about a single ICD10 code."""
    index: int = Field(..., description="Index of the code in the model")
    code: str = Field(..., description="ICD10 code")
    label: str = Field(..., description="Human-readable label for the condition")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status", example="ok")


app = FastAPI(
    title="Delphi Inference API",
    description="API for predicting future medical conditions using the Delphi model based on patient history",
    version="1.0.0",
    openapi_url="/api/openapi.json",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# CORS configuration - always allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=False,  # Must be False when using wildcard
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)


# Global state loaded at startup
STATE: Dict[str, Any] = {
    "device": None,
    "model": None,
    "labels_dict": None,
    "code_to_index": None,
    "index_to_code": None,
}


def encode_patient_sequence(patient_data, code_to_index):
    encoded_tokens = []
    encoded_ages = []
    for code, age in patient_data:
        if code in code_to_index:
            token_idx = code_to_index[code] + 1  # reserve 0 for padding
            age_days = age * 365.25
            encoded_tokens.append(token_idx)
            encoded_ages.append(age_days)

    return (torch.tensor(encoded_tokens, dtype=torch.long),
            torch.tensor(encoded_ages, dtype=torch.float32))


def extract_label_text(raw_text: str) -> str:
    """Extract human-readable label without code prefix.
    Examples:
        "A06 (amoebiasis)" -> "amoebiasis"
        "E66 (obesity)"    -> "obesity"
        Fallbacks to removing code prefix if parentheses not present.
    """
    label_match = re.match(r"^[A-Z]\d{2}\s*\((.*)\)\s*$", raw_text)
    if label_match:
        return label_match.group(1)
    fallback = re.sub(r"^[A-Z]\d{2}\s*", "", raw_text).strip()
    return fallback if fallback else raw_text


@app.on_event("startup")
def load_model_and_mappings():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = os.path.join(REPO_ROOT, "out", "ckpt.pt")
    if not os.path.isfile(ckpt_path):
        raise RuntimeError(f"Checkpoint not found at {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    gptconf = DelphiConfig(**checkpoint["model_args"])  # type: ignore
    model = Delphi(gptconf)
    model.load_state_dict(checkpoint["model"])  # type: ignore
    model.to(device)
    model.eval()

    labels_path = os.path.join(REPO_ROOT, "data", "ukb_simulated_data", "labels.csv")
    if not os.path.isfile(labels_path):
        raise RuntimeError(f"Labels file not found at {labels_path}")

    labels_dict: Dict[int, str] = {}
    code_to_index: Dict[str, int] = {}
    with open(labels_path, "r") as f:
        for i, line in enumerate(f):
            text = line.strip()
            labels_dict[i] = text
            m = re.match(r"^([A-Z]\d{2})", text)
            if m:
                code_to_index[m.group(1)] = i

    index_to_code = {v: k for k, v in code_to_index.items()}

    STATE["device"] = device
    STATE["model"] = model
    STATE["labels_dict"] = labels_dict
    STATE["code_to_index"] = code_to_index
    STATE["index_to_code"] = index_to_code


@app.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Health check",
    description="Check if the API service is running and healthy",
    tags=["Health"],
)
def health() -> HealthResponse:
    """Health check endpoint to verify the API is operational."""
    return HealthResponse(status="ok")


@app.post(
    "/predict",
    response_model=PredictResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict future conditions",
    description="Predict the top 10 most likely future medical conditions based on a patient's history of ICD10 codes and ages. Optionally specify a timeframe in years to get predictions for events within that period.",
    tags=["Prediction"],
    responses={
        200: {
            "description": "Successful prediction",
            "model": PredictResponse,
        },
        400: {
            "description": "Bad request - no valid ICD10 codes after filtering",
        },
        503: {
            "description": "Service unavailable - model not loaded",
        },
    },
)
def predict(req: PredictRequest) -> PredictResponse:
    model: Optional[Delphi] = STATE.get("model")
    device = STATE.get("device")
    code_to_index: Dict[str, int] = STATE.get("code_to_index") or {}
    labels_dict: Dict[int, str] = STATE.get("labels_dict") or {}
    index_to_code: Dict[int, str] = STATE.get("index_to_code") or {}

    if model is None or device is None or not code_to_index:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Filter unknown codes; collect warnings
    input_pairs = [(e.code, float(e.age_at_event)) for e in req.patient]
    filtered = [(c, a) for c, a in input_pairs if c in code_to_index]
    warnings = []
    if len(filtered) != len(input_pairs):
        missing = [c for c, _ in input_pairs if c not in code_to_index]
        warnings.append(f"Removed codes not found in labels: {missing}")

    if not filtered:
        raise HTTPException(status_code=400, detail="No valid ICD10 codes after filtering")

    tokens, ages = encode_patient_sequence(filtered, code_to_index)
    tokens = tokens.unsqueeze(0).to(device)
    ages = ages.unsqueeze(0).to(device)

    with torch.no_grad():
        # If timeframe is specified, use generate method to get predictions within that timeframe
        if req.timeframe_years is not None and req.timeframe_years > 0:
            # Calculate max_age: current patient age + timeframe
            current_age_days = ages[0, -1].item()  # Last age in days
            max_age_days = current_age_days + (req.timeframe_years * 365.25)
            
            # Use generate to get predictions within the timeframe
            # Generate a trajectory up to max_age, then extract probabilities
            generated_idx, generated_age, generated_logits = model.generate(
                tokens,
                ages,
                max_new_tokens=100,  # Generate up to 100 new events
                max_age=max_age_days,
                no_repeat=True,
                termination_tokens=None,
                top_k=None
            )
            
            # Get probabilities from the final logits of the generated sequence
            # Average across all positions in the generated sequence to get aggregate probabilities
            final_logits = generated_logits[:, -1, :].squeeze()
            probs = torch.softmax(final_logits, dim=-1)
        else:
            # Default behavior: single forward pass for immediate next event
            logits, _, _ = model(tokens, ages)
            last_logits = logits[:, -1, :].squeeze()
            probs = torch.softmax(last_logits, dim=-1)

    probs_list = probs.tolist()
    sorted_indices = sorted(range(len(probs_list)), key=lambda i: probs_list[i], reverse=True)

    ranked = []
    for i in sorted_indices:
        code = index_to_code.get(i)
        raw_label = labels_dict.get(i)
        label = extract_label_text(raw_label) if raw_label is not None else None
        ranked.append({
            "index": i,
            "code": code if code is not None else None,
            "label": label,
            "probability": float(probs_list[i]),
        })

    return PredictResponse(
        ranked=[RankedPrediction(**item) for item in ranked[:10]],
        used_patient=[PatientEvent(code=c, age_at_event=a) for c, a in filtered],
        warnings=warnings,
    )

@app.get(
    "/codes",
    response_model=List[CodeInfo],
    status_code=status.HTTP_200_OK,
    summary="List available ICD10 codes",
    description="Get a list of all available ICD10 codes that the model can predict",
    tags=["Codes"],
    responses={
        200: {
            "description": "List of available ICD10 codes",
            "model": List[CodeInfo],
        },
        503: {
            "description": "Service unavailable - labels not loaded",
        },
    },
)
def list_codes() -> List[CodeInfo]:
    labels_dict: Dict[int, str] = STATE.get("labels_dict") or {}
    if not labels_dict:
        raise HTTPException(status_code=503, detail="Labels not loaded")

    items = []
    for i in sorted(labels_dict.keys()):
        text = labels_dict[i]
        m = re.match(r"^([A-Z]\d{2})", text)
        # Skip entries that don't have a valid ICD10 code prefix
        if not m:
            continue
        code = m.group(1)
        label_text = extract_label_text(text)
        items.append(CodeInfo(
            index=i,
            code=code,
            label=label_text,
        ))
    return items


@app.get(
    "/api",
    response_class=JSONResponse,
    summary="OpenAPI specification",
    description="Get the OpenAPI 3.0 specification for this API",
    tags=["Documentation"],
)
def get_openapi_spec() -> Dict[str, Any]:
    """Return the OpenAPI specification as JSON."""
    return app.openapi()


