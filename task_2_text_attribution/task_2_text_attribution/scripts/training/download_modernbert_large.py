from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

# Scratch-Verzeichnis
SCRATCH_DIR = "/p/scratch/training2562/zainzinger1"
os.environ["HF_HOME"] = os.path.join(SCRATCH_DIR, "hf_cache")
os.environ["TMPDIR"] = os.path.join(SCRATCH_DIR, "tmp")

MODELS_DIR = os.path.join(SCRATCH_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

print("=== LADE MODERNBERT-LARGE ===")
model_name = "answerdotai/ModernBERT-large"
model_path = os.path.join(MODELS_DIR, "modernbert-large")

print("Lade Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.save_pretrained(model_path)

print("Lade Modell...")
model = AutoModelForSequenceClassification.from_pretrained(model_name, trust_remote_code=True)
model.save_pretrained(model_path)

print(f"=== FERTIG! Gespeichert in {model_path} ===")
