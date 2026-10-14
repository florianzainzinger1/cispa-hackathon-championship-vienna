"""
TF-IDF Classifier - "Cheap Boost" Strategy
Fängt Vokabular-Artefakte von LLMs (z.B. "delve", "tapestry")
Training dauert <2 Minuten, kann +0.01 Score bringen!
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import joblib
import os

# --- CONFIG ---
TRAIN_FILE = "participants_train.csv"
TEST_FILE = "participants_test.csv"
MODEL_FILE = "tfidf_model.joblib"
OUTPUT_FILE = "probs_tfidf.csv"

def main():
    print("=== TF-IDF CLASSIFIER ===")
    
    # Lade Training-Daten
    df = pd.read_csv(TRAIN_FILE)
    df = df.dropna(subset=["response"])
    df["response"] = df["response"].astype(str)
    
    labels = sorted(df['model'].unique().tolist())
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}
    df['label'] = df['model'].map(label2id)
    
    print(f"Klassen: {len(labels)}")
    print(f"Samples: {len(df)}")
    
    # Train/Val Split
    X_train, X_val, y_train, y_val = train_test_split(
        df["response"], df["label"], test_size=0.1, stratify=df["label"], random_state=42
    )
    
    # TF-IDF Vectorizer (n-grams 1-3)
    print("Fitting TF-IDF (ngrams 1-3)...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=50000,
        min_df=2,
        sublinear_tf=True
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)
    
    print(f"Features: {X_train_tfidf.shape[1]}")
    
    # Logistic Regression
    print("Training Logistic Regression...")
    clf = LogisticRegression(
        max_iter=1000,
        n_jobs=-1,
        C=1.0,
        class_weight='balanced'
    )
    clf.fit(X_train_tfidf, y_train)
    
    # Validation Score
    y_pred = clf.predict(X_val_tfidf)
    val_f1 = f1_score(y_val, y_pred, average='macro')
    print(f"\nValidation Macro-F1: {val_f1:.4f}")
    
    # Test Predictions
    print("\nMaking test predictions...")
    test_df = pd.read_csv(TEST_FILE)
    test_df["response"] = test_df["response"].fillna("").astype(str)
    
    X_test_tfidf = vectorizer.transform(test_df["response"])
    probs = clf.predict_proba(X_test_tfidf)
    
    # Speichern
    probs_df = pd.DataFrame(probs, columns=[id2label[i] for i in range(len(labels))])
    probs_df["uuid"] = test_df["uuid"].values
    probs_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved: {OUTPUT_FILE}")
    
    # Submission
    predictions = [id2label[i] for i in clf.predict(X_test_tfidf)]
    submission = pd.DataFrame({"uuid": test_df["uuid"], "model": predictions})
    submission.to_csv("submission_tfidf.csv", index=False)
    print(f"Saved: submission_tfidf.csv")
    
    # Model speichern
    joblib.dump({"vectorizer": vectorizer, "classifier": clf, "id2label": id2label}, MODEL_FILE)
    print(f"Model saved: {MODEL_FILE}")

if __name__ == "__main__":
    main()
