#!/usr/bin/env python3
"""
Quick test script to verify the dataset inference pipeline works.
Runs on a small subset of data to catch errors before full job submission.
"""

import os
import sys

# Change to the correct directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("Dataset Inference Pipeline - Quick Test")
print("=" * 60)

# Test 1: Check imports
print("\n[1/5] Testing imports...")
try:
    import torch
    import pandas as pd
    import numpy as np
    from scipy import stats
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler
    import torchvision.models as models
    print("  ✓ All imports successful")
except ImportError as e:
    print(f"  ✗ Import error: {e}")
    sys.exit(1)

# Test 2: Check files exist
print("\n[2/5] Checking required files...")
required_files = ['subsets_dataset.pt', 'classifier.pt']
for f in required_files:
    if os.path.exists(f):
        size_mb = os.path.getsize(f) / (1024 * 1024)
        print(f"  ✓ {f} ({size_mb:.1f} MB)")
    else:
        print(f"  ✗ {f} not found!")
        sys.exit(1)

# Test 3: Load model
print("\n[3/5] Loading model...")
try:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")
    
    model = models.resnet18(weights=None)
    model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=3, bias=False)
    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2),
        torch.nn.Linear(model.fc.in_features, 9)
    )
    model.load_state_dict(torch.load('classifier.pt', map_location='cpu'))
    model.to(device)
    model.eval()
    print("  ✓ Model loaded successfully")
except Exception as e:
    print(f"  ✗ Model loading error: {e}")
    sys.exit(1)

# Test 4: Load dataset and check format
print("\n[4/5] Loading and checking dataset...")
try:
    dataset = torch.load('subsets_dataset.pt')
    print(f"  Number of subsets: {len(dataset)}")
    
    # Check first subset
    subset_0 = dataset['subset_0']
    print(f"  Subset 0 keys: {list(subset_0.keys())}")
    print(f"  Images shape: {subset_0['images'].shape}")
    print(f"  Labels shape: {subset_0['labels'].shape}")
    print(f"  Subset ID: {subset_0['subset_id']}")
    print("  ✓ Dataset format verified")
except Exception as e:
    print(f"  ✗ Dataset error: {e}")
    sys.exit(1)

# Test 5: Run inference on one subset
print("\n[5/5] Testing inference on subset 0...")
try:
    images = subset_0['images'].to(device)
    labels = subset_0['labels'].to(device)
    
    with torch.no_grad():
        logits = model(images)
    
    print(f"  Logits shape: {logits.shape}")
    print(f"  Logits range: [{logits.min().item():.3f}, {logits.max().item():.3f}]")
    
    # Quick feature extraction test
    import torch.nn.functional as F
    nll = F.cross_entropy(logits, labels, reduction='none')
    print(f"  NLL mean: {nll.mean().item():.4f}")
    print(f"  NLL std: {nll.std().item():.4f}")
    
    probs = F.softmax(logits, dim=1)
    entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=1)
    print(f"  Entropy mean: {entropy.mean().item():.4f}")
    
    print("  ✓ Inference successful")
except Exception as e:
    print(f"  ✗ Inference error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("All tests passed! Ready to run full pipeline.")
print("=" * 60)
print("\nNext steps:")
print("  1. Run dry-run: python dataset_inference.py --dry-run")
print("  2. Submit full job: sbatch run_inference.sh")
print("")
