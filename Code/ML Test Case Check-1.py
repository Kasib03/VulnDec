import numpy as np
import joblib
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics import classification_report, confusion_matrix

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
# ✅ Test Code Snippets + Expected Labels
# ---------------------------
test_snippets = [
    "cursor.execute('SELECT * FROM users WHERE id=' + user_input)",               # 0
    "<script>alert(document.cookie)</script>",                                    # 1
    "<form action='/delete' method='POST'><input type='submit' value='Delete'>",  # 2
    "__import__('os').system('ls')",                                              # 3
    "redirect_to = request.GET.get('next')\nreturn redirect(redirect_to)",        # 4
    "print(e)  # error contains full traceback with file path",                   # 5
    "os.system('ping ' + ip_address)",                                            # 6
    "print('Welcome back, user!')",                                               # -1
    "def add(x, y): return x + y",                                                # -1
    "for user in users:\n    print(user.name)"                                    # -1
]

expected_classes = [
    0, 1, 2, 3, 4, 5, 6, -1, -1, -1
]

# ---------------------------
# ✅ Predict Function
# ---------------------------
def predict_all(snippets, expected=None):
    print("\n=== Predictions on 10 Realistic Cases ===")
    
    # --- Random Forest
    pred_rf = clf_rf.predict(tfidf.transform(snippets))
    print("\n[Random Forest + TF-IDF]")
    
    # --- Logistic Regression
    pred_lr = clf_lr.predict(bow.transform(snippets))
    print("\n[Logistic Regression + BoW]")
    
    # --- CodeBERT Embeddings
    print("\n[Computing CodeBERT embeddings...]")
    codebert_avg = np.array([get_avg_codebert_embedding(code) for code in snippets])
    
    # --- SVM
    pred_svm = clf_svm.predict(scaler.transform(codebert_avg))
    print("\n[SVM + CodeBERT]")
    
    # --- XGBoost
    pred_xgb = clf_xgb.predict(codebert_avg)
    print("\n[XGBoost + CodeBERT]")

    # --- Results Table
    print("\n--- Results ---")
    for i, code in enumerate(snippets):
        print(f"\n🔹 Snippet {i+1}: {code[:50]}...")
        true_label = "Clean" if expected[i] == -1 else CLASS_NAMES[expected[i]]
        print(f"   ✅ Expected: {true_label}")
        print(f"   🔍 RF  → {CLASS_NAMES[pred_rf[i]]}")
        print(f"   🔍 LR  → {CLASS_NAMES[pred_lr[i]]}")
        print(f"   🔍 SVM → {CLASS_NAMES[pred_svm[i]]}")
        print(f"   🔍 XGB → {CLASS_NAMES[pred_xgb[i]]}")

    # --- Accuracy on labeled (non-clean) examples
    mask = [i for i, label in enumerate(expected) if label != -1]
    y_true = [expected[i] for i in mask]
    print("\n--- Accuracy on Vulnerability Samples ---")
    print(f"Random Forest  : {np.mean([pred_rf[i] == expected[i] for i in mask]):.2f}")
    print(f"Logistic Reg.  : {np.mean([pred_lr[i] == expected[i] for i in mask]):.2f}")
    print(f"SVM            : {np.mean([pred_svm[i] == expected[i] for i in mask]):.2f}")
    print(f"XGBoost        : {np.mean([pred_xgb[i] == expected[i] for i in mask]):.2f}")

# ---------------------------
# ✅ Run Test
# ---------------------------
predict_all(test_snippets, expected_classes)
