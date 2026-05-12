import os
import zipfile
import numpy as np
from pathlib import Path
from PIL import Image, ImageFilter
import random

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms
import torchvision.transforms.functional

# ----------------------------
# CONFIG
# ----------------------------
ZIP_FILE = "Dataset.zip"       # Path to dataset zip
DATASET_DIR = Path("Dataset")  # Unzipped folder
SUBMISSION_FILE = "submission.npz"

# Leaderboard submission
# Leaderboard submission
SERVER_URL = "http://35.192.205.84:80"
API_KEY = None  # teams insert their assigned token here
TASK_ID = "09-watermark-removal"


# ----------------------------
# UNZIP DATASET
# ----------------------------
if not DATASET_DIR.exists():
    print("Unzipping dataset...")
    with zipfile.ZipFile(ZIP_FILE, "r") as zip_ref:
        zip_ref.extractall(".")
else:
    print("Dataset already extracted.")


# ----------------------------
# CUSTOM DATASET
# ----------------------------
transform_to_tensor = torchvision.transforms.ToTensor()

class TestDataset(Dataset):
    def __init__(self, root):
        self.root = Path(root)
        self.files = sorted(self.root.glob("*.png"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        img = Image.open(path).convert("RGB")
        img_tensor = transform_to_tensor(img)

        return img_tensor, path.name

dataset = TestDataset(DATASET_DIR)
test_loader = DataLoader(dataset, batch_size=8, shuffle=False)


# ----------------------------
# DUMMY MODEL
# ----------------------------
def dummy_model(img_pil):
    """Apply a simple blur as a placeholder."""
    return img_pil.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))


# ----------------------------
# BUILD SUBMISSION
# ----------------------------
print("Building submission...")

watermarked_dir = DATASET_DIR
test_dataset = TestDataset(watermarked_dir)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

if len(test_dataset) != 100:
    raise ValueError(f"Expected 100 images for submission, found {len(test_dataset)}.")

outputs, names = [], []

for images_batch, names_batch in test_loader:
    for img_tensor, name in zip(images_batch, names_batch):
        img_pil = torchvision.transforms.functional.to_pil_image(img_tensor)

        cleaned = dummy_model(img_pil)

        outputs.append(np.array(cleaned))
        names.append(int(os.path.splitext(name)[0]))


np.savez_compressed(SUBMISSION_FILE, images=outputs, names=names)
print(f"Saved submission file to {SUBMISSION_FILE}")
print("   Format: np.savez_compressed('submission.npz', images=array, names=array)")


# ----------------------------
# SUBMIT TO LEADERBOARD SERVER
# ----------------------------
if API_KEY is None:
    print("No TOKEN provided. Please set your team TOKEN in this script to submit.")
else:
    print("Submitting to leaderboard server...")

    response = requests.post(
        f"{SERVER_URL}/submit/{TASK_ID}",
        files={"file": open(SUBMISSION_FILE, "rb")},
        headers={"X-API-Key": API_KEY},
    )
    print("Server response:", response.json())


