import os
from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from fastapi.middleware.cors import CORSMiddleware

# ========================
# PATH DINAMIS ABSOLUT
# ========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # backend/
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))  # bert-ai-project
MODEL_DIR = os.path.join(ROOT_DIR, "models", "sentiment_indobert")
MODEL_DIR = os.path.abspath(MODEL_DIR)  # wajib absolut

device = "cuda" if torch.cuda.is_available() else "cpu"
print(">>> Inference running on:", device)

# ========================
# APP & CORS
# ========================
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ========================
# INPUT SCHEMA
# ========================
class InputText(BaseModel):
    text: str

# ========================
# LOAD MODEL & TOKENIZER
# ========================
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, local_files_only=True)
model.to(device)
model.eval()

labels = ["Negative", "Neutral", "Positive"]

# ========================
# ENDPOINT PREDIKSI
# ========================
@app.post("/predict")
def predict(payload: InputText):
    encoded = tokenizer(payload.text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        logits = model(**encoded).logits
        probs = F.softmax(logits, dim=-1)[0].cpu().numpy()

    pred = int(probs.argmax())
    return {
        "sentiment": labels[pred],
        "confidence": float(probs[pred]),
        "probs": probs.tolist()
    }
