"""
MASTER ENSEMBLE - Combines ALL Winning Strategies
1. Length-Aware Weighting (DeBERTa-Large for short, ModernBERT-Large for long)
2. TF-IDF Boost (adds 10% probability from simple TF-IDF model)
3. Entropy-Based Outlier Detection (filters unknown models)
"""
import pandas as pd
import numpy as np
import os
from scipy.stats import entropy

# --- CONFIG ---
DEBERTA_PROBS = "probs_deberta_large.csv"
MODERNBERT_PROBS = "probs_modernbert_large.csv"
TFIDF_PROBS = "probs_tfidf.csv"
OUTPUT_FILE = "submission_master_ensemble.csv"

# 1. Length-Aware Weights
SHORT_TEXT_THRESHOLD = 400  # words
# Short texts: DeBERTa is better at subtle style
SHORT_WEIGHT_DEBERTA = 0.50
SHORT_WEIGHT_MODERNBERT = 0.40
SHORT_WEIGHT_TFIDF = 0.10

# Long texts: ModernBERT is king (8k context)
LONG_WEIGHT_DEBERTA = 0.00
LONG_WEIGHT_MODERNBERT = 0.90
LONG_WEIGHT_TFIDF = 0.10

# 3. Entropy Threshold
ENTROPY_THRESHOLD = 2.0  # Tuned value

def calculate_entropy(probs):
    probs = np.clip(probs, 1e-10, 1)
    return entropy(probs)

def main():
    print("=== MASTER ENSEMBLE STRATEGY ===")
    
    # Check files
    for f in [DEBERTA_PROBS, MODERNBERT_PROBS, TFIDF_PROBS]:
        if not os.path.exists(f):
            print(f"WAITING: {f} not found yet.")
            return

    print("Loading probabilities...")
    df_deb = pd.read_csv(DEBERTA_PROBS)
    df_mb = pd.read_csv(MODERNBERT_PROBS)
    df_tfidf = pd.read_csv(TFIDF_PROBS)
    
    # Load test data for length
    test_df = pd.read_csv("participants_test.csv")
    test_df["word_count"] = test_df["response"].astype(str).str.split().str.len()
    
    # Columns
    uuids = df_deb["uuid"]
    exclude = ["uuid", "max_prob"]
    cols = [c for c in df_deb.columns if c not in exclude]
    
    predictions = []
    outliers = 0
    
    print(f"Processing {len(uuids)} samples...")
    
    for i in range(len(uuids)):
        # Get probs
        p_deb = df_deb[cols].iloc[i].values
        p_mb = df_mb[cols].iloc[i].values
        p_tfidf = df_tfidf[cols].iloc[i].values
        
        # Determine weights based on length
        wc = test_df["word_count"].iloc[i]
        
        if wc > SHORT_TEXT_THRESHOLD:
            # Long text strategy
            final_p = (LONG_WEIGHT_DEBERTA * p_deb + 
                      LONG_WEIGHT_MODERNBERT * p_mb + 
                      LONG_WEIGHT_TFIDF * p_tfidf)
        else:
            # Short text strategy
            final_p = (SHORT_WEIGHT_DEBERTA * p_deb + 
                      SHORT_WEIGHT_MODERNBERT * p_mb + 
                      SHORT_WEIGHT_TFIDF * p_tfidf)
            
        # Normalize
        final_p = final_p / final_p.sum()
        
        # Entropy Check
        ent = calculate_entropy(final_p)
        
        if ent > ENTROPY_THRESHOLD:
            predictions.append("outlier")
            outliers += 1
        else:
            predictions.append(cols[final_p.argmax()])
            
    # Save
    sub = pd.DataFrame({"uuid": uuids, "model": predictions})
    sub.to_csv(OUTPUT_FILE, index=False)
    
    print(f"\n=== DONE! Saved {OUTPUT_FILE} ===")
    print(f"Outliers detected: {outliers} ({outliers/len(predictions)*100:.2f}%)")
    print("\nTop predicted classes:")
    print(sub["model"].value_counts().head())

if __name__ == "__main__":
    main()
