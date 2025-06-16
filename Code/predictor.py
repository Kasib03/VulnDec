import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ---------------- CONFIG ----------------
MODEL_PATH = "model_multiclass_cnn_bilstm_codebert_blocks_2klimit_kaggle.keras"
CODEBERT_MODEL = "microsoft/codebert-base"
MAX_LEN = 200

label_map = [
    "sql", "xss", "xsrf", "remote_code_execution",
    "open_redirect", "path_disclosure", "command_injection"
]

# ---------------- LOAD MODEL + CODEBERT ----------------
print("[INFO] Loading trained CNN+BiLSTM model...")
model = load_model(MODEL_PATH)

print("[INFO] Loading CodeBERT...")
tokenizer = AutoTokenizer.from_pretrained(CODEBERT_MODEL)
codebert = AutoModel.from_pretrained(CODEBERT_MODEL)
codebert = codebert.cuda() if torch.cuda.is_available() else codebert
codebert.eval()

# ---------------- UTILITIES ----------------
def stripComments(code):
    lines = code.split("\n")
    cleaned = ""
    for line in lines:
        if "#" in line:
            line = line[:line.find("#")]
        cleaned += line + "\n"
    return cleaned

def get_codebert_embedding(text):
    cleaned = stripComments(text)
    inputs = tokenizer(cleaned, return_tensors="pt", truncation=True, padding="max_length", max_length=MAX_LEN)
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    with torch.no_grad():
        outputs = codebert(**inputs)
    return outputs.last_hidden_state.squeeze(0).cpu().numpy()

# ---------------- PREDICT FUNCTION ----------------
def predict_vulnerability(code_snippet):
    emb = get_codebert_embedding(code_snippet)
    emb_padded = pad_sequences([emb], maxlen=MAX_LEN, dtype='float32', padding='post', truncating='post')
    
    prediction = model.predict(emb_padded)
    predicted_label = np.argmax(prediction)
    confidence = prediction[0][predicted_label]
    
    label_name = label_map[predicted_label]
    print(f"\n🛡️ Predicted Vulnerability: {label_name.upper()} (class {predicted_label})")
    print(f"🔍 Confidence: {confidence:.2f}")
    if confidence < 0.6:
        print("⚠️ Low confidence. Manual review suggested.")

# ---------------- TEST CASES ----------------
if __name__ == "__main__":
    test_cases = {
        # Standard
        "SQL Injection": '''
username = request.args.get("username")
cursor.execute("SELECT * FROM users WHERE username = '" + username + "'")
''',
        "XSS": '''
comment = request.args.get("comment")
return "<html>" + comment + "</html>"
''',
        "XSRF": '''
@app.route("/update_password", methods=["POST"])
def update_password():
    password = request.form['new_password']
    update_user_password(current_user, password)
''',
        "Remote Code Execution": '''
@app.route('/run')
def run():
    cmd = request.args.get('cmd')
    return subprocess.check_output(cmd, shell=True)
''',
        "Open Redirect": '''
@app.route('/redirect')
def go():
    url = request.args.get('next')
    return redirect(url)
''',
        "Path Disclosure": '''
try:
    with open("user_data/" + user_id + ".json") as f:
        data = json.load(f)
except Exception as e:
    return str(e)
''',
        "Command Injection": '''
os.system("rm -rf " + request.args.get("dir"))
''',

        # Advanced / obfuscated
        "SQLi (obfuscated)": '''
user = input("Username:")
sql = f"SELECT * FROM login WHERE user = '{user}'"
execute_query(sql)
''',
        "XSS (reflected)": '''
@app.route("/")
def home():
    name = request.args.get("name")
    return "<p>Hi " + name + "</p>"
''',
        "XSRF (no protection)": '''
@app.route("/settings", methods=["POST"])
def settings():
    user = current_user
    theme = request.form['theme']
    update_theme(user, theme)
''',
        "RCE (via eval)": '''
cmd = request.args.get("c")
eval(cmd)
''',
        "Redirect (dynamic)": '''
@app.route("/nav")
def nav():
    target = request.args.get("to")
    return redirect(target)
''',
        "Path Disclosure (stack trace)": '''
try:
    do_something_sensitive()
except Exception as e:
    return traceback.format_exc()
''',
        "CMD Inject (env chain)": '''
command = request.args.get("x")
os.system("echo $PATH && " + command)
'''
    }

    for i, (name, code) in enumerate(test_cases.items(), 1):
        print(f"\n=== 🔍 Test Case {i}: {name} ===")
        predict_vulnerability(code)
