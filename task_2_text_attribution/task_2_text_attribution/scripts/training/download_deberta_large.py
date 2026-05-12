from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

# Scratch-Verzeichnis
SCRATCH_DIR = "/p/scratch/training2562/zainzinger1"
os.environ["HF_HOME"] = os.path.join(SCRATCH_DIR, "hf_cache")
os.environ["TMPDIR"] = os.path.join(SCRATCH_DIR, "tmp")

MODELS_DIR = os.path.join(SCRATCH_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

print("=== LADE DEBERTA-V3-LARGE ===")
model_name = "microsoft/deberta-v3-large"
model_path = os.path.join(MODELS_DIR, "deberta-v3-large")

print("Lade Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(model_path)

print("Lade Modell...")
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.save_pretrained(model_path)

print(f"=== FERTIG! Gespeichert in {model_path} ===")
