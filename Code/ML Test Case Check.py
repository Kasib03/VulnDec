import numpy as np
import joblib
from transformers import AutoTokenizer, AutoModel
import torch

# ---------------------------
# ✅ Load Saved Models
# ---------------------------
tfidf = joblib.load("vectorizer_tfidf.joblib")
bow = joblib.load("vectorizer_bow.joblib")
scaler = joblib.load("scaler_codebert.joblib")

clf_rf = joblib.load("model_random_forest.joblib")
clf_lr = joblib.load("model_logistic_regression.joblib")
clf_svm = joblib.load("model_svm.joblib")
clf_xgb = joblib.load("model_xgboost.joblib")

# ---------------------------
# ✅ Load CodeBERT (for SVM/XGBoost)
# ---------------------------
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
codebert = AutoModel.from_pretrained("microsoft/codebert-base")
codebert.eval()

def get_avg_codebert_embedding(text, max_length=200):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=max_length)
    with torch.no_grad():
        outputs = codebert(**inputs)
    embedding = outputs.last_hidden_state.squeeze(0).numpy()
    return np.mean(embedding, axis=0)

# ---------------------------
# ✅ Class Label Mapping
# ---------------------------
CLASS_NAMES = [
    "SQL Injection", "Cross-Site Scripting (XSS)", "Cross-Site Request Forgery (XSRF)",
    "Remote Code Execution", "Open Redirect", "Path Disclosure", "Command Injection"
]

# ---------------------------
# ✅ Test Code Snippets
# ---------------------------
test_snippets = [
    "import os\nos.system('rm -rf /')",  # likely command injection
    "print('Hello World')",              # clean
    "cursor.execute('SELECT * FROM users WHERE id=' + user_input)",  # likely SQLi
]

# ---------------------------
# ✅ Predict Function
# ---------------------------
def predict_all(snippets):
    print("\n=== Predictions ===")
    
    # --- Random Forest
    tfidf_features = tfidf.transform(snippets)
    pred_rf = clf_rf.predict(tfidf_features)
    print("\n[Random Forest + TF-IDF]")
    for i, pred in enumerate(pred_rf):
        print(f"Snippet {i+1}: {CLASS_NAMES[pred]}")

    # --- Logistic Regression
    bow_features = bow.transform(snippets)
    pred_lr = clf_lr.predict(bow_features)
    print("\n[Logistic Regression + BoW]")
    for i, pred in enumerate(pred_lr):
        print(f"Snippet {i+1}: {CLASS_NAMES[pred]}")

    # --- CodeBERT Embeddings
    print("\n[Computing CodeBERT embeddings...]")
    codebert_avg = np.array([get_avg_codebert_embedding(code) for code in snippets])
    
    # --- SVM
    svm_scaled = scaler.transform(codebert_avg)
    pred_svm = clf_svm.predict(svm_scaled)
    print("\n[SVM + CodeBERT]")
    for i, pred in enumerate(pred_svm):
        print(f"Snippet {i+1}: {CLASS_NAMES[pred]}")

    # --- XGBoost
    pred_xgb = clf_xgb.predict(codebert_avg)
    print("\n[XGBoost + CodeBERT]")
    for i, pred in enumerate(pred_xgb):
        print(f"Snippet {i+1}: {CLASS_NAMES[pred]}")

# ---------------------------
# ✅ Run Predictions
# ---------------------------
predict_all(test_snippets)
