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
from model import Delphi, DelphiConfig

# ----------------------------
# Helper function to encode patient sequence
# ----------------------------
def encode_patient_sequence(patient_data, code_to_index, time_window=24):
    """
    Encode a patient sequence into the format expected by the model.

    Args:
        patient_data: List of tuples (ICD10_code, age_at_event)
        code_to_index: Dictionary mapping ICD10 codes to indices
        time_window: Maximum sequence length

    Returns:
        torch.Tensor: Encoded sequence of shape (sequence_length,)
    """
    # Convert codes to indices and ages to days
    encoded_tokens = []
    encoded_ages = []

    for code, age in patient_data:
        if code in code_to_index:
            # Add 1 to token index (0 is reserved for padding)
            token_idx = code_to_index[code] + 1
            # Convert age to days
            age_days = age * 365.25
            encoded_tokens.append(token_idx)
            encoded_ages.append(age_days)

    # Pad or truncate to time_window
    if len(encoded_tokens) > time_window:
        encoded_tokens = encoded_tokens[:time_window]
        encoded_ages = encoded_ages[:time_window]
    else:
        # Pad with zeros
        while len(encoded_tokens) < time_window:
            encoded_tokens.append(0)  # padding token
            encoded_ages.append(-10000)  # mask time

    return torch.tensor(encoded_tokens, dtype=torch.long), torch.tensor(encoded_ages, dtype=torch.float32)

# ----------------------------
# CONFIG
# ----------------------------
OUTPUT_DIR = os.path.join(repo_root, "outputs")
MODEL_PATH = os.path.join(OUTPUT_DIR, "ckpt.pt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Create a synthetic patient timeline
# ----------------------------
# Format: list of tuples (ICD10 code, age_at_event)
# Note: We'll filter this to only include codes that exist in the labels file
synthetic_patient = [
    ("J06", 66),
    ("E66", 35),
    ("I10", 42),
    ("M10", 35),
    ("N03", 12),
    ("N11", 45),
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
# Load the actual disease labels and generate code_to_index mapping
# ----------------------------
import pandas as pd
import re

# Load the labels file to get the actual disease names and generate code_to_index mapping
labels_dict = {}
code_to_index = {}

with open(os.path.join(repo_root, "data/ukb_simulated_data/labels.csv"), 'r') as f:
    for i, line in enumerate(f):
        labels_dict[i] = line.strip()

        # Extract ICD10 code from the line (format: "CODE (description)")
        # Look for pattern like "E66 (obesity)" or "I10 (essential (primary) hypertension)"
        match = re.match(r'^([A-Z]\d{2})', line.strip())
        if match:
            icd10_code = match.group(1)
            code_to_index[icd10_code] = i

print(f"Generated code_to_index mapping with {len(code_to_index)} ICD10 codes")
print(f"Available codes for synthetic patient: {[code for code in ['Z00', 'J06', 'E66', 'I10'] if code in code_to_index]}")

# Filter synthetic patient data to only include codes that exist in the labels
filtered_patient = [(code, age) for code, age in synthetic_patient if code in code_to_index]
if len(filtered_patient) != len(synthetic_patient):
    missing_codes = [code for code, age in synthetic_patient if code not in code_to_index]
    print(f"⚠️  Removed codes not found in labels: {missing_codes}")
    print(f"Using filtered patient data: {filtered_patient}")

# Create index_to_code mapping
index_to_code = {v: k for k, v in code_to_index.items()}

# Add model attributes that the script expects
model.code_to_index = code_to_index
model.index_to_code = index_to_code

encoded_tokens, encoded_ages = encode_patient_sequence(
    filtered_patient,
    code_to_index=code_to_index,
    time_window=24  # Use a reasonable time window
)
encoded_tokens = encoded_tokens.unsqueeze(0).to(DEVICE)  # batch size 1
encoded_ages = encoded_ages.unsqueeze(0).to(DEVICE)  # batch size 1

# ----------------------------
# Run inference
# ----------------------------
with torch.no_grad():
    logits, _, _ = model(encoded_tokens, encoded_ages)

# Convert logits to probabilities
# Get the last position logits (next token prediction)
last_logits = logits[:, -1, :].squeeze()  # Shape: (vocab_size,)
probs = torch.softmax(last_logits, dim=-1)
probs_list = probs.tolist()

# Sort indices by descending probability
sorted_indices = sorted(range(len(probs_list)), key=lambda i: probs_list[i], reverse=True)

# ----------------------------
# Print results
# ----------------------------
print("\n🧠 Top 5 predicted future conditions for synthetic patient:")
for i in sorted_indices[:5]:
    if i in labels_dict:
        disease_name = labels_dict[i]
        print(f"• {disease_name:<50} : {probs_list[i]*100:.2f}%")
    else:
        print(f"• Token_{i:<45} : {probs_list[i]*100:.2f}%")

print("\n✅ Simulation complete.")
