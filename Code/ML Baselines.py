# ML Baselines for Code Vulnerability Detection (Local Execution)

# ✅ Install Dependencies (uncomment if needed)


import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ✅ Load Preprocessed Data from Pickle Files
print("Loading X_filtered.pkl, y_filtered.pkl, raw_code.pkl...")

with open("X_filtered.pkl", "rb") as f:
    X_filtered = pickle.load(f)

with open("y_filtered.pkl", "rb") as f:
    y_filtered = pickle.load(f)

with open("raw_code.pkl", "rb") as f:
    raw_code = pickle.load(f)

# ✅ Random Forest + TF-IDF
print("\n=== Random Forest + TF-IDF ===")
tfidf = TfidfVectorizer(max_features=2000)
X_tfidf = tfidf.fit_transform(raw_code)
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y_filtered, test_size=0.2, random_state=42)
clf_rf = RandomForestClassifier(n_estimators=100, class_weight="balanced")
clf_rf.fit(X_train, y_train)
y_pred = clf_rf.predict(X_test)
print(classification_report(y_test, y_pred))

# ✅ Logistic Regression + BoW
print("\n=== Logistic Regression + BoW ===")
bow = CountVectorizer(max_features=2000)
X_bow = bow.fit_transform(raw_code)
X_train, X_test, y_train, y_test = train_test_split(X_bow, y_filtered, test_size=0.2, random_state=42)
clf_lr = LogisticRegression(max_iter=1000, class_weight='balanced')
clf_lr.fit(X_train, y_train)
y_pred = clf_lr.predict(X_test)
print(classification_report(y_test, y_pred))

# ✅ SVM + Avg CodeBERT Embedding
print("\n=== SVM + Avg CodeBERT Embedding ===")
X_avg = np.array([np.mean(x, axis=0) for x in X_filtered])
X_train, X_test, y_train, y_test = train_test_split(X_avg, y_filtered, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
clf_svm = SVC(kernel='rbf', class_weight='balanced')
clf_svm.fit(X_train_scaled, y_train)
y_pred = clf_svm.predict(X_test_scaled)
print(classification_report(y_test, y_pred))

# ✅ XGBoost + Avg CodeBERT Embedding
print("\n=== XGBoost + Avg CodeBERT Embedding ===")
X_train, X_test, y_train, y_test = train_test_split(X_avg, y_filtered, test_size=0.2, random_state=42)
clf_xgb = XGBClassifier(objective='multi:softmax', num_class=7, eval_metric='mlogloss', use_label_encoder=False)
clf_xgb.fit(X_train, y_train)
y_pred = clf_xgb.predict(X_test)
print(classification_report(y_test, y_pred))


import joblib  # More robust for saving sklearn models than pickle

# 🔹 Save Vectorizers
joblib.dump(tfidf, "vectorizer_tfidf.joblib")
joblib.dump(bow, "vectorizer_bow.joblib")
joblib.dump(scaler, "scaler_codebert.joblib")

# 🔹 Save Models
joblib.dump(clf_rf, "model_random_forest.joblib")
joblib.dump(clf_lr, "model_logistic_regression.joblib")
joblib.dump(clf_svm, "model_svm.joblib")
joblib.dump(clf_xgb, "model_xgboost.joblib")

print("✅ All models and preprocessors saved!")
