# cnn_bilstm_codebert_multiclass.py (optimized for local use, 2000 samples per file)

import os
import json
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, AutoModel
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
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
EMBED_DIM = 768  # CodeBERT
BATCH_SIZE = 8
EPOCHS = 5
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

# -------- LOAD AND EMBED DATA (LIMIT TO 2000) --------
X, y = [], []

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
                        count += 1
                        if count >= 2000:
                            break
                    except Exception as e:
                        print(f"Embedding failed: {e}")
                if count >= 2000:
                    break
            if count >= 2000:
                break
        if count >= 2000:
            break

# -------- FILTER LABELED SAMPLES --------
X_filtered = [X[i] for i in range(len(X)) if y[i] != -1]
y_filtered = [y[i] for i in range(len(y)) if y[i] != -1]

X_pad = pad_sequences(X_filtered, maxlen=MAX_LEN, dtype='float32', padding='post', truncating='post')
y_cat = to_categorical(y_filtered, num_classes=len(MODELS))

# -------- SPLIT DATA --------
X_train, X_temp, y_train, y_temp = train_test_split(X_pad, y_cat, test_size=0.3, random_state=42, stratify=y_cat)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# -------- CNN + BiLSTM MODEL --------
print("Building CNN+BiLSTM model...")
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=5, activation='relu', input_shape=(MAX_LEN, EMBED_DIM)))
model.add(MaxPooling1D(pool_size=2))
model.add(Bidirectional(LSTM(64, dropout=0.3)))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(len(MODELS), activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# -------- TRAIN --------
print("Training...")
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=EPOCHS, batch_size=BATCH_SIZE)

# -------- EVALUATE --------
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(y_test, axis=1)

print("Classification Report:")
print(classification_report(y_true_labels, y_pred_labels))

# -------- SAVE MODEL --------
model.save("model_multiclass_cnn_bilstm_codebert_blocks_2klimit.keras")
print("Model saved as model_multiclass_cnn_bilstm_codebert_blocks_2klimit.keras")
