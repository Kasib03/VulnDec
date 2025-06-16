# cnn_bilstm_codebert_multiclass.py (enhanced with raw code export for ML baselines)

import os
import json
import torch
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoTokenizer, AutoModel
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns
import myutils

# -------- CONFIG --------
MODELS = [
    ("plain_sql.json", 0),
    ("plain_xss.json", 1),
    ("plain_xsrf.json", 2),
    ("plain_remote_code_execution.json", 3),
    ("plain_open_redirect.json", 4),
    ("plain_path_disclosure.json", 5),
    ("plain_command_injection.json", 6)
]

MAX_LEN = 200
EMBED_DIM = 768
BATCH_SIZE = 8
EPOCHS = 15
MODEL_NAME = "microsoft/codebert-base"
STEP = 5

# -------- LOAD CODEBERT --------
print("Loading CodeBERT...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model = model.cuda() if torch.cuda.is_available() else model
model.eval()

def get_codebert_embedding(text):
    text = myutils.stripComments(text)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=MAX_LEN)
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.squeeze(0).cpu().numpy()

# -------- LOAD AND EMBED DATA --------
X, y = [], []
raw_code_all = []

for filename, label in MODELS:
    print(f"Loading: {filename} (label {label})")
    if not os.path.isfile(filename):
        print(f"[ERROR] File not found: {filename}")
        continue
    with open(filename, 'r') as f:
        data = json.load(f)

    count = 0
    for repo in data:
        for commit in data[repo]:
            files = data[repo][commit].get("files", {})
            for fname, filedata in files.items():
                if "source" not in filedata:
                    continue
                sourcecode = filedata["source"]
                allbadparts = []
                for change in filedata.get("changes", []):
                    parts = myutils.getBadpart(change.get("diff", ""))
                    if parts is not None:
                        allbadparts.extend(parts[0])
                badpositions = myutils.findpositions(allbadparts, sourcecode)
                blocks = myutils.getblocks(sourcecode, badpositions, STEP, MAX_LEN)
                for block in blocks:
                    try:
                        emb = get_codebert_embedding(block[0])
                        X.append(emb)
                        y.append(label if block[1] == 1 else -1)
                        raw_code_all.append(block[0])  # ✅ Save raw snippet
                        count += 1
                        if count >= 5000:
                            break
                    except Exception as e:
                        print(f"Embedding failed: {e}")
                if count >= 5000:
                    break
            if count >= 5000:
                break
        if count >= 5000:
            break

# -------- FILTER VALID ENTRIES --------
X_filtered = [X[i] for i in range(len(X)) if y[i] != -1]
y_filtered = [y[i] for i in range(len(y)) if y[i] != -1]
raw_code_filtered = [raw_code_all[i] for i in range(len(y)) if y[i] != -1]

# -------- SAVE DATA FOR CLASSICAL ML MODELS --------
with open("X_filtered.pkl", "wb") as f:
    pickle.dump(X_filtered, f)

with open("y_filtered.pkl", "wb") as f:
    pickle.dump(y_filtered, f)

with open("raw_code.pkl", "wb") as f:
    pickle.dump(raw_code_filtered, f)

print("✅ Saved: X_filtered.pkl, y_filtered.pkl, raw_code.pkl")
