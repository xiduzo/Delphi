import os
import re
import sys
from typing import List, Optional, Dict, Any

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure repo root is importable
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from model import Delphi, DelphiConfig  # noqa: E402


class PatientEvent(BaseModel):
    code: str = Field(..., description="ICD10 code, e.g. 'I10'")
    age_at_event: float = Field(..., description="Age in years at the event")


class PredictRequest(BaseModel):
    patient: List[PatientEvent]


app = FastAPI(title="Delphi Inference API")

# CORS configuration to allow frontend requests during local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
    ckpt_path = os.path.join(REPO_ROOT, "delphi", "ckpt.pt")
    if not os.path.isfile(ckpt_path):
        raise RuntimeError(f"Checkpoint not found at {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)
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


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
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

    return {
        "ranked": ranked[:10],
        "used_patient": [{"code": c, "age_at_event": a} for c, a in filtered],
        "warnings": warnings,
    }

@app.get("/codes")
def list_codes():
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
        items.append({
            "index": i,
            "code": code,
            "label": label_text,
        })
    return items


