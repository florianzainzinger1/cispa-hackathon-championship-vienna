"""
Length-Aware Ensemble - Paper-basierte Strategie
- Lange Texte (>512 tokens): Trust ModernBERT (sieht alles)
- Kurze Texte (<512 tokens): Trust DeBERTa (bessere Semantik)
- Outlier Detection: DeBERTa-Unsicherheit nutzen
"""
import pandas as pd
import numpy as np
import os

# --- CONFIG ---
SCRATCH_DIR = "/p/scratch/training2562/zainzinger1"

# Probability Files
PROBS_FILES = {
    "deberta": "probs_deberta.csv",
    "modernbert": "probs_modernbert.csv",
    "deberta_large": "probs_deberta_large.csv",
    "modernbert_large": "probs_modernbert_large.csv"
}

OUTPUT_FILE = "submission_length_aware.csv"

# Token-Grenzen (Wörter als Proxy für Tokens, ~1.3 tokens per word)
SHORT_TEXT_THRESHOLD = 400  # ~512 tokens
LONG_TEXT_THRESHOLD = 400   # Dateipfade
DEBERTA_PROBS = "probs_deberta_large.csv"  # UPGRADE: Use Large!
MODERNBERT_PROBS = "probs_modernbert_large.csv"  # UPGRADE: Use Large!

# Gewichtungen für kurze Texte (<400 Wörter)
SHORT_WEIGHT_DEBERTA = 0.55
SHORT_WEIGHT_MODERNBERT = 0.45

# Gewichtungen für lange Texte (>400 Wörter) - Paper: 100% ModernBERT!
LONG_WEIGHT_DEBERTA = 0.0
LONG_WEIGHT_MODERNBERT = 1.0

# Outlier Detection: Wenn max(prob) < threshold UND DeBERTa unsicher
OUTLIER_THRESHOLD = 0.35  # Set to None to disable
DEBERTA_UNCERTAINTY_WEIGHT = 0.7  # DeBERTa's uncertainty zählt mehr für Outliers

def main():
    print("=== LENGTH-AWARE ENSEMBLE ===")
    print(f"Short texts (<{SHORT_TEXT_THRESHOLD} words): DeBERTa={SHORT_WEIGHT_DEBERTA}, ModernBERT={SHORT_WEIGHT_MODERNBERT}")
    print(f"Long texts  (>{LONG_TEXT_THRESHOLD} words): DeBERTa={LONG_WEIGHT_DEBERTA}, ModernBERT={LONG_WEIGHT_MODERNBERT}")
    
    # Lade Test-Daten für Textlängen
    test_df = pd.read_csv("participants_test.csv")
    test_df["word_count"] = test_df["response"].astype(str).str.split().str.len()
    
    # Stats
    short_count = (test_df["word_count"] <= SHORT_TEXT_THRESHOLD).sum()
    long_count = (test_df["word_count"] > LONG_TEXT_THRESHOLD).sum()
    print(f"\nTexte: {short_count} kurz, {long_count} lang")
    
    # Finde beste verfügbare Modelle
    available = {}
    for name, path in PROBS_FILES.items():
        if os.path.exists(path):
            available[name] = pd.read_csv(path)
            print(f"✓ {name} geladen")
    
    # Priorisiere Large-Modelle falls verfügbar
    deberta_key = "deberta_large" if "deberta_large" in available else "deberta"
    modernbert_key = "modernbert_large" if "modernbert_large" in available else "modernbert"
    
    if deberta_key not in available or modernbert_key not in available:
        print("FEHLER: Brauche mindestens je ein DeBERTa und ModernBERT Modell!")
        return
    
    print(f"\nVerwende: {deberta_key} + {modernbert_key}")
    
    df_deb = available[deberta_key]
    df_mod = available[modernbert_key]
    
    uuids = df_deb["uuid"]
    exclude_cols = ["uuid", "max_prob"]
    prob_cols = [c for c in df_deb.columns if c not in exclude_cols]
    
    probs_deb = df_deb[prob_cols].values
    probs_mod = df_mod[prob_cols].values
    
    # Length-Aware Ensemble
    predictions = []
    outlier_count = 0
    
    for i, (uuid, word_count) in enumerate(zip(uuids, test_df["word_count"])):
        # Wähle Gewichte basierend auf Textlänge
        if word_count <= SHORT_TEXT_THRESHOLD:
            w_deb, w_mod = SHORT_WEIGHT_DEBERTA, SHORT_WEIGHT_MODERNBERT
        else:
            w_deb, w_mod = LONG_WEIGHT_DEBERTA, LONG_WEIGHT_MODERNBERT
        
        # Gewichteter Durchschnitt
        prob = w_deb * probs_deb[i] + w_mod * probs_mod[i]
        
        max_prob = prob.max()
        
        # Outlier Detection mit DeBERTa-Unsicherheit
        if OUTLIER_THRESHOLD:
            deb_max_prob = probs_deb[i].max()
            # Gewichtete Unsicherheit: DeBERTa zählt mehr
            combined_uncertainty = DEBERTA_UNCERTAINTY_WEIGHT * (1 - deb_max_prob) + \
                                   (1 - DEBERTA_UNCERTAINTY_WEIGHT) * (1 - probs_mod[i].max())
            
            if max_prob < OUTLIER_THRESHOLD and combined_uncertainty > 0.6:
                predictions.append("outlier")
                outlier_count += 1
                continue
        
        predictions.append(prob_cols[prob.argmax()])
    
    print(f"\nOutliers: {outlier_count} ({outlier_count/len(predictions)*100:.2f}%)")
    
    # Submission erstellen
    submission = pd.DataFrame({"uuid": uuids, "model": predictions})
    submission.to_csv(OUTPUT_FILE, index=False)
    print(f"\n=== Gespeichert: {OUTPUT_FILE} ===")
    print(submission["model"].value_counts().head(10))

if __name__ == "__main__":
    main()
