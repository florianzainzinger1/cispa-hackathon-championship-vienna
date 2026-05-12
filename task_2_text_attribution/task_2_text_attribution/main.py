import torch as th
import pandas as pd
import requests
import time
import sys
import os

# --------------------------------
# DATASET
# --------------------------------

print("LLM attribution")

"""
Dataset contents:

- 84700 samples of text.
  Each sample is a dictionary with:
    -"model": The name of the LLM that generated the text.
    -"temperature": The temperature setting used during generation.
    -"top_p": The top_p setting used during generation.
    -"domain": The domain/category of the prompt.
    -"response": The generated text response.
"""

print("Load the train dataset.")
dataset = pd.read_csv("participants_train.csv")

# Example: Accessing the first sample
first_sample = dataset.iloc[0]
print("First sample:", first_sample)

# --------------------------------
# SUBMISSION FORMAT
# --------------------------------

"""
The submission must be a .csv file with the following format:

-"uuid": Unique identifier for each text sample, this is given in the participants_test.csv file (string)
-"model": The predicted model name that generated the text (string)
"""

# Example Submission:

print("Load the test dataset.")
dataset = pd.read_csv("participants_test.csv")

uuids = dataset["uuid"].tolist()
model_name = "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
models = [model_name for _ in range(len(uuids))]
submission_df = pd.DataFrame({
    "uuid": uuids,
    "model": models
})
submission_df.to_csv("submission.csv", index=None)

# --------------------------------
# SUBMISSION PROCESS
# --------------------------------

API_KEY  = os.environ.get("HACKATHON_API_KEY") 

TASK_ID  = "07-llm-attribution"
FILE_PATH = "Your-Submission-File.csv" # <- Path to your real submission file
BASE_URL = "http://35.192.205.84:80"

SUBMIT     = False  # Set to True to enable submission

def die(msg):
    print(f"{msg}", file=sys.stderr)
    sys.exit(1)

if SUBMIT:
    if not os.path.isfile(FILE_PATH):
        die(f"File not found: {FILE_PATH}")

    try:
        with open(FILE_PATH, "rb") as f:
            files = {
                # (fieldname) -> (filename, fileobj, content_type)
                "file": (os.path.basename(FILE_PATH), f, "csv"),
            }
            resp = requests.post(
                f"{BASE_URL}/submit/{TASK_ID}",
                headers={"X-API-Key": API_KEY},
                files=files,
                timeout=(10, 120),  # (connect timeout, read timeout)
            )
        # Helpful output even on non-2xx
        try:
            body = resp.json()
        except Exception:
            body = {"raw_text": resp.text}

        if resp.status_code == 413:
            die("Upload rejected: file too large (HTTP 413). Reduce size and try again.")

        resp.raise_for_status()

        submission_id = body.get("submission_id")
        print("Successfully submitted.")
        print("Server response:", body)
        if submission_id:
            print(f"Submission ID: {submission_id}")

    except requests.exceptions.RequestException as e:
        detail = getattr(e, "response", None)
        print(f"Submission error: {e}")
        if detail is not None:
            try:
                print("Server response:", detail.json())
            except Exception:
                print("Server response (text):", detail.text)
        sys.exit(1)
