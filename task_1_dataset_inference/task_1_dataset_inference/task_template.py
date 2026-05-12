import torch
import pandas as pd
import requests
import sys
import torchvision.models as models
import os

# --------------------------------
# DATASET
# --------------------------------

"""
Dataset contents:

- 1000 subsets of images, each subset stored under the key "subset_{i}" where i ranges from 0 to 999.
  Each subset is a dictionary with:
    -"images": Tensor containing the 100 images in the subset, has shape (100, 1, 28, 28)
    -"labels": Tensor of true labels for the images in the subset, has shape (100)
    -"subset_id": Integer ID of the subset (from 0 to 999)
"""

# Load the dataset
dataset = torch.load("subsets_dataset.pt")

# Example: Acessing subsets
subset_0 = dataset["subset_0"]

print("Subset 0 keys:", subset_0.keys())
print("Subset ID:", subset_0["subset_id"])
print("Images shape:", subset_0["images"].shape)
print("Labels shape:", subset_0["labels"].shape) 
print("First image tensor:", subset_0["images"][:1])
print("First 10 labels:", subset_0["labels"][:10])

# --------------------------------
# QUERYING THE CLASSIFIER
# --------------------------------

# You can use the following Code to load and query the image classifier with images:

model = models.resnet18(weights=None)
model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=3, bias=False)
model.fc = torch.nn.Sequential(
    torch.nn.Dropout(p=0.2),
    torch.nn.Linear(model.fc.in_features, 9)
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.load_state_dict(torch.load("classifier.pt", map_location="cpu"))
model.to(device)

images = subset_0["images"]

output = model(images)

print("Logits shape:", output.shape)  # Should be (100, 9) for subset of 100 images and 9 classes
print("First 5 logits:", output[:5]) 

# --------------------------------
# SUBMISSION FORMAT
# --------------------------------

"""
The submission must be a .csv file with the following format:

-"subset_id": ID of the subset (from 0 to 999)
-"membership": Membership score for each image (float)
"""

# Example Submission:

subset_ids = list(range(len(dataset)))  
membership_scores = torch.rand(len(dataset)).tolist()
submission_df = pd.DataFrame({
    "subset_id": subset_ids,
    "membership": membership_scores
})
submission_df.to_csv("example_submission.csv", index=None)

# --------------------------------
# SUBMISSION PROCESS
# --------------------------------

"""
Example submission script for the Vision Set Membership Inference Task.

Submission Requirements (read carefully to avoid automatic rejection):

1. CSV FORMAT
----------------
- The file **must be a CSV** with extension `.csv`.
- It must contain **exactly two columns**, named:
      subset_id, membership
  → Column names must match exactly (lowercase, no extra spaces).
  → Column order does not matter, but both must be present.

2. ROW COUNT AND IDENTIFIERS
-------------------------------
- Your file must contain **exactly 1,000 rows**.
- Each row corresponds to one unique `subset_id` in the range **0–999** (inclusive).
- Every subset_id must appear **exactly once**.
- Do **not** add, remove, or rename any IDs.
- Do **not** include duplicates or missing entries.
- The evaluator checks:
      subset_id.min() == 0
      subset_id.max() == 999
      subset_id.unique().size == 1000

3. MEMBERSHIP SCORES
----------------------
- The `membership` column must contain **numeric values** representing your model’s predicted confidence
  that the corresponding subset is a **member** of the training set.

  Examples of valid membership values:
    - Probabilities: values in [0.0, 1.0]
    - Raw model scores: any finite numeric values (will be ranked for TPR@FPR=0.05)

- Do **not** submit string labels like "yes"/"no" or "member"/"non-member".
- The evaluator converts your `membership` column to numeric using `pd.to_numeric()`.
  → Any non-numeric, NaN, or infinite entries will cause automatic rejection.

4. TECHNICAL LIMITS
----------------------
- Maximum file size: **20 MB**
- Encoding: UTF-8 recommended.
- Avoid extra columns, blank lines, or formulas.
- Ensure all values are numeric and finite.
- Supported data types: int, float (e.g., float32, float64)

5. VALIDATION SUMMARY
------------------------
Your submission will fail if:
- Columns don’t match exactly ("subset_id", "membership")
- Row count differs from 1,000
- Any subset_id is missing, duplicated, or outside [0, 999]
- Any membership value is NaN, Inf, or non-numeric
- File is too large or not a valid CSV

Two key metrics are computed:
  1. **ROC-AUC (Area Under the ROC Curve)** — measures overall discriminative ability.
  2. **TPR@FPR=0.05** — true positive rate when the false positive rate is at 5%.

"""

BASE_URL  = "http://35.192.205.84:80"
API_KEY   = "YOUR_API_KEY_HERE"  # replace with your actual API key

TASK_ID   = "06-dataset-inference-vision"
FILE_PATH = "Your-Submission-File.csv"  # replace with your actual file path

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