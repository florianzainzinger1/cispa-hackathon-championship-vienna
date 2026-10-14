"""
Predict-Skript für Text Attribution (mit Outlier-Detection)
Lädt ein trainiertes Modell und macht Vorhersagen auf den Testdaten.
"""
import pandas as pd
import numpy as np
import torch
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

# --- CONFIG ---
SCRATCH_DIR = "/p/scratch/training2562/zainzinger1"

# Wähle welches Modell verwendet werden soll:
MODEL_TYPE = "deberta_large"  # "deberta", "modernbert", oder "deberta_large"

MODEL_PATHS = {
    "deberta": os.path.join(SCRATCH_DIR, "model_deberta_trained"),
    "modernbert": os.path.join(SCRATCH_DIR, "model_modernbert_trained"),
    "deberta_large": os.path.join(SCRATCH_DIR, "model_deberta_large_trained"),
    "modernbert_large": os.path.join(SCRATCH_DIR, "model_modernbert_large_trained")
}

MODEL_PATH = MODEL_PATHS[MODEL_TYPE]

TEST_FILE = "participants_test.csv"
OUTPUT_FILE = f"submission_{MODEL_TYPE}.csv"
BATCH_SIZE = 32
MAX_LEN = 512

# --- OUTLIER DETECTION ---
# Set to None to disable, or a float (e.g., 0.4) to enable
# If max probability < threshold, assign "outlier"
OUTLIER_THRESHOLD = None  # Will be set from best_threshold.txt if exists

def main():
    global OUTLIER_THRESHOLD
    
    # Load threshold from file if exists
    if os.path.exists("best_threshold.txt") and OUTLIER_THRESHOLD is None:
        with open("best_threshold.txt", "r") as f:
            OUTLIER_THRESHOLD = float(f.read().strip())
        print(f"Loaded outlier threshold from file: {OUTLIER_THRESHOLD}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model: {MODEL_PATH}")
    print(f"Outlier Threshold: {OUTLIER_THRESHOLD}")
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    
    # Modell & Tokenizer laden
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model.to(device)
    model.eval()
    
    # Label-Mapping aus dem Modell
    id2label = model.config.id2label
    print(f"Klassen: {list(id2label.values())}")
    
    # Testdaten laden
    print(f"Lade Testdaten: {TEST_FILE}")
    df = pd.read_csv(TEST_FILE)
    df["response"] = df["response"].fillna("").astype(str)
    
    print(f"Anzahl Testbeispiele: {len(df)}")
    
    # Vorhersagen machen
    predictions = []
    probabilities = []
    max_probs = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(df), BATCH_SIZE), desc="Vorhersage"):
            batch_texts = df["response"].iloc[i:i+BATCH_SIZE].tolist()
            
            # Tokenisieren
            inputs = tokenizer(
                batch_texts, 
                truncation=True, 
                max_length=MAX_LEN, 
                padding=True, 
                return_tensors="pt"
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Vorhersage
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            
            probabilities.extend(probs.cpu().numpy())
            max_probs.extend(probs.max(dim=-1).values.cpu().numpy())
    
    # Apply outlier threshold if set
    for i, prob in enumerate(probabilities):
        if OUTLIER_THRESHOLD and prob.max() < OUTLIER_THRESHOLD:
            predictions.append("outlier")
        else:
            pred_idx = prob.argmax()
            predictions.append(id2label[pred_idx])
    
    # Statistics
    n_outliers = sum(1 for p in predictions if p == "outlier")
    print(f"\nOutlier predictions: {n_outliers} ({n_outliers/len(predictions)*100:.2f}%)")
    
    # Confidence stats
    max_probs = np.array(max_probs)
    print(f"Confidence - Min: {max_probs.min():.3f}, Max: {max_probs.max():.3f}, Mean: {max_probs.mean():.3f}")
    
    # Labels zuordnen
    df["model"] = predictions
    
    # Submission erstellen
    submission = df[["uuid", "model"]]
    submission.to_csv(OUTPUT_FILE, index=False)
    print(f"\n=== Submission gespeichert: {OUTPUT_FILE} ===")
    print(submission["model"].value_counts().head(10))
    
    # Auch Wahrscheinlichkeiten speichern (für Ensemble)
    probs_df = pd.DataFrame(probabilities, columns=[id2label[i] for i in range(len(id2label))])
    probs_df["uuid"] = df["uuid"].values
    probs_df["max_prob"] = max_probs  # For outlier analysis
    probs_df.to_csv(f"probs_{MODEL_TYPE}.csv", index=False)
    print(f"Wahrscheinlichkeiten gespeichert: probs_{MODEL_TYPE}.csv")

if __name__ == "__main__":
    main()
