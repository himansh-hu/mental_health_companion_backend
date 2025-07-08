# Mental‑Health Companion — Backend

Flask + TensorFlow API that:

1. Predicts the user’s emotional tone from text (`/predict`).
2. Generates a compassionate reply using Meta Llama 3.1 8B (GitHub Models API).
3. Optionally logs conversations to Firebase Firestore.

---

##  Stack

| Layer | Tech |
|-------|------|
| REST API | **Flask** & Gunicorn |
| ML Model | TensorFlow `.keras` + `scikit‑learn` tokenizer / label encoder |
| LLM | GitHub Models API (`meta/Meta-Llama-3.1-8B-Instruct`) via **`azure.ai.inference`** SDK |
| Cloud DB | Firebase Firestore |
| Deploy | Render Web Service |

---

##  Environment variables

Create a real **`.env`** (not committed) based on **`.env.example`**:

| Key | Description |
|-----|-------------|
| `GITHUB_TOKEN` | GitHub PAT with **`models:read`** scope |
| `MODEL_SLUG` | `meta/Meta-Llama-3.1-8B-Instruct` |
| `AZURE_OPENAI_ENDPOINT` | `https://models.github.ai/inference` |
| `GOOGLE_APPLICATION_CREDENTIALS` | Path to your Firebase Admin JSON |

---

##  Local Run

```bash
# 1  Create and activate venv
python -m venv .venv
source .venv/bin/activate

# 2  Install deps
pip install -r requirements.txt

# 3  Copy & fill .env
cp .env.example .env

# 4  Launch dev server on all interfaces
python flask_api.py
# ⇒ http://127.0.0.1:5000  and  http://<LAN_IP>:5000
