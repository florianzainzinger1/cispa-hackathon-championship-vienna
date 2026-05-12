"""
Submit-Skript für Text Attribution
Sendet die Submission an den Leaderboard-Server.
"""
import requests
import os
import sys

# --- CONFIG ---
API_KEY = "5091102ba147d8bece3d901377dbb6d1"
TASK_ID = "07-llm-attribution"
BASE_URL = "http://35.192.205.84:80"

# Option: "deberta", "modernbert", oder "ensemble"
# Option: "deberta", "modernbert", oder "ensemble"
SUBMISSION_TYPE = "deberta_large_v5"

FILE_PATHS = {
    "deberta": "submission_deberta.csv",
    "deberta_large": "submission_deberta_large.csv",
    "deberta_large_v3": "submission_deberta_large_v3.csv",
    "modernbert": "submission_modernbert.csv",
    "tta": "submission_modernbert_tta.csv",
    "modernbert_base_v6": "submission_modernbert_base_v6.csv",
    "modernbert_large_v5": "submission_modernbert_large_v5.csv",
    "voting_ensemble": "submission_voting_ensemble.csv",
    "avg_ensemble": "submission_avg_ensemble.csv",
    "swapped": "submission_swapped.csv",
    "length_corrected": "submission_length_corrected.csv",
    "deberta_large_v5": "submission_deberta_large_v5.csv",
    "3model_ensemble": "submission_3model_ensemble.csv",
    "best_ensemble": "submission_best_ensemble.csv",
    "confidence_weighted": "submission_confidence_weighted.csv",
    "soft_voting": "submission_soft_voting.csv",
    "max_conf": "submission_max_conf.csv", 
    "ensemble": "submission_ensemble.csv",
    "length_aware": "submission_length_aware.csv",
    "modernbert_large": "submission_modernbert_large.csv",
    "modernbert_large_v3": "submission_modernbert_large_v3_corrected.csv",
    "mb_large_tfidf": "submission_mb_large_tfidf.csv",
    "mb_large_entropy": "submission_mb_large_entropy.csv",
    "stacking": "submission_stacking.csv",
    "master": "submission_master_ensemble.csv",
    "pure_large": "submission_pure_large_ensemble.csv",
    "tta": "submission_tta.csv"
}

def submit(file_path):
    if not os.path.isfile(file_path):
        print(f"FEHLER: {file_path} nicht gefunden!")
        sys.exit(1)
    
    print(f"Sende {file_path} an Server...")
    
    with open(file_path, "rb") as f:
        files = {"file": (os.path.basename(file_path), f, "csv")}
        resp = requests.post(
            f"{BASE_URL}/submit/{TASK_ID}",
            headers={"X-API-Key": API_KEY},
            files=files,
            timeout=(10, 120),
        )
    
    try:
        body = resp.json()
    except:
        body = {"raw_text": resp.text}
    
    if resp.status_code == 413:
        print("FEHLER: Datei zu groß!")
        sys.exit(1)
    
    resp.raise_for_status()
    
    print("\n=== ERFOLGREICH! ===")
    print(f"Server Antwort: {body}")
    if "submission_id" in body:
        print(f"Submission ID: {body['submission_id']}")

def main():
    if API_KEY == "YOUR_API_KEY_HERE":
        print("FEHLER: Bitte API_KEY im Skript eintragen!")
        sys.exit(1)
    
    file_path = FILE_PATHS.get(SUBMISSION_TYPE)
    if not file_path:
        print(f"FEHLER: Unbekannter SUBMISSION_TYPE: {SUBMISSION_TYPE}")
        sys.exit(1)
    
    submit(file_path)

if __name__ == "__main__":
    main()
