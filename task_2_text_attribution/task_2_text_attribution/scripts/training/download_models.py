from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

# WICHTIG: Alles auf Scratch setzen (Home hat zu wenig Platz!)
SCRATCH_DIR = "/p/scratch/training2562/zainzinger1"

# HuggingFace Cache
os.environ["HF_HOME"] = os.path.join(SCRATCH_DIR, "hf_cache")
# Temp-Verzeichnis für Downloads
os.environ["TMPDIR"] = os.path.join(SCRATCH_DIR, "tmp")
os.environ["TEMP"] = os.environ["TMPDIR"]
os.environ["TMP"] = os.environ["TMPDIR"]

os.makedirs(os.environ["HF_HOME"], exist_ok=True)
os.makedirs(os.environ["TMPDIR"], exist_ok=True)

MODELS_DIR = os.path.join(SCRATCH_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

print(f"HF Cache: {os.environ['HF_HOME']}")
print(f"TMPDIR: {os.environ['TMPDIR']}")
print(f"Models Dir: {MODELS_DIR}")

# DeBERTa ist schon da, also überspringen
deberta_path = os.path.join(MODELS_DIR, "deberta-v3-base")
if os.path.exists(os.path.join(deberta_path, "model.safetensors")):
    print("\n--- DeBERTa bereits vorhanden, überspringe... ---")
else:
    print("\n--- 1. Lade DeBERTa herunter... ---")
    deberta_name = "microsoft/deberta-v3-base"
    tokenizer = AutoTokenizer.from_pretrained(deberta_name)
    tokenizer.save_pretrained(deberta_path)
    model = AutoModelForSequenceClassification.from_pretrained(deberta_name)
    model.save_pretrained(deberta_path)
    print(f"DeBERTa gespeichert in {deberta_path}")

print("\n--- 2. Lade ModernBERT herunter... ---")
modern_name = "answerdotai/ModernBERT-base"
modern_path = os.path.join(MODELS_DIR, "modernbert-base")

tokenizer = AutoTokenizer.from_pretrained(modern_name, trust_remote_code=True)
tokenizer.save_pretrained(modern_path)
print("Tokenizer gespeichert!")

model = AutoModelForSequenceClassification.from_pretrained(modern_name, trust_remote_code=True)
model.save_pretrained(modern_path)
print(f"ModernBERT gespeichert in {modern_path}")

print("\n=== FERTIG! ===")