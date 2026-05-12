#!/usr/bin/env python3
"""
Submit the generated submission.csv to the leaderboard.
"""

import os
import sys
import argparse
import requests

# ================================
# CONFIGURATION - UPDATE THIS!
# ================================
API_KEY = "5091102ba147d8bece3d901377dbb6d1"  # Team: !!1337
# ================================

BASE_URL = "http://35.192.205.84:80"
TASK_ID = "06-dataset-inference-vision"
DEFAULT_FILE_PATH = "submission.csv"

def main():
    parser = argparse.ArgumentParser(description="Submit to leaderboard")
    parser.add_argument("--submission", type=str, default=DEFAULT_FILE_PATH,
                        help="Path to submission CSV file")
    args = parser.parse_args()
    FILE_PATH = args.submission
    if API_KEY == "YOUR_API_KEY_HERE":
        print("ERROR: Please set your API_KEY in submit.py!")
        print("Edit the file and replace 'YOUR_API_KEY_HERE' with your team token.")
        sys.exit(1)
    
    if not os.path.isfile(FILE_PATH):
        print(f"ERROR: File not found: {FILE_PATH}")
        sys.exit(1)
    
    print(f"Submitting {FILE_PATH} to leaderboard...")
    print(f"  Task: {TASK_ID}")
    print(f"  Server: {BASE_URL}")
    
    try:
        with open(FILE_PATH, "rb") as f:
            files = {
                "file": (os.path.basename(FILE_PATH), f, "text/csv"),
            }
            resp = requests.post(
                f"{BASE_URL}/submit/{TASK_ID}",
                headers={"X-API-Key": API_KEY},
                files=files,
                timeout=(10, 120),
            )
        
        try:
            body = resp.json()
        except Exception:
            body = {"raw_text": resp.text}
        
        if resp.status_code == 413:
            print("ERROR: Upload rejected - file too large (HTTP 413)")
            sys.exit(1)
        
        resp.raise_for_status()
        
        print("\n" + "="*50)
        print("SUCCESS! Submission accepted.")
        print("="*50)
        print(f"Server response: {body}")
        
        if "submission_id" in body:
            print(f"Submission ID: {body['submission_id']}")
        if "score" in body:
            print(f"Score: {body['score']}")
        if "tpr_at_fpr_005" in body:
            print(f"TPR@FPR=0.05: {body['tpr_at_fpr_005']}")
            
    except requests.exceptions.RequestException as e:
        print(f"Submission error: {e}")
        if hasattr(e, 'response') and e.response is not None:
            try:
                print("Server response:", e.response.json())
            except:
                print("Server response (text):", e.response.text)
        sys.exit(1)

if __name__ == "__main__":
    main()
