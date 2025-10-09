# simulate_person_inference.py
"""
Simulate a synthetic patient's health timeline using the Delphi trained model.

This script:
- Fixes module import issues by adding the repo root to sys.path
- Loads the trained checkpoint from delphi/ckpt.pt
- Creates a synthetic patient timeline
- Runs inference and prints predicted future diseases
"""

import sys
import os
import torch

# ----------------------------
# Fix import path for Delphi package
# ----------------------------
repo_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(repo_root)

# ----------------------------
# Imports from Delphi
# ----------------------------
from delphi.model import Delphi, DelphiConfig
from delphi.data.data_utils import encode_patient_sequence

# ----------------------------
# CONFIG
# ----------------------------
OUTPUT_DIR = os.path.join(repo_root, "delphi")
MODEL_PATH = os.path.join(OUTPUT_DIR, "ckpt.pt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Create a synthetic patient timeline
# ----------------------------
# Format: list of tuples (ICD10 code, age_at_event)
synthetic_patient = [
    ("Z00", 25),  # general checkup
    ("J06", 27),  # respiratory infection
    ("E66", 35),  # obesity
    ("I10", 42),  # hypertension
]

# ----------------------------
# Load the trained Delphi model
# ----------------------------
print("🔹 Loading trained Delphi checkpoint...")
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
gptconf = DelphiConfig(**checkpoint['model_args'])
model = Delphi(gptconf)
model.load_state_dict(checkpoint['model'])
model.to(DEVICE)
model.eval()

# ----------------------------
# Encode the patient sequence
# ----------------------------
encoded_sequence = encode_patient_sequence(
    synthetic_patient,
    code_to_index=model.code_to_index,
    time_window=model.cfg.model.time_window
)
encoded_sequence = encoded_sequence.unsqueeze(0).to(DEVICE)  # batch size 1

# ----------------------------
# Run inference
# ----------------------------
with torch.no_grad():
    logits, _, _ = model(encoded_sequence)

# Convert logits to probabilities
probs = torch.softmax(logits, dim=-1).squeeze().tolist()
# Sort indices by descending probability
sorted_indices = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)

# ----------------------------
# Print results
# ----------------------------
print("\n🧠 Top 5 predicted future conditions for synthetic patient:")
for i in sorted_indices[:5]:
    code = model.index_to_code[i]
    print(f"• {code:<10} : {probs[i]*100:.2f}%")

print("\n✅ Simulation complete.")
