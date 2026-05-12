#!/usr/bin/env python3
"""
Dataset Inference Attack - Best Approach (Score: 0.200)
========================================================

This implementation achieves TPR@FPR=5% of 0.200 using:
- Extreme value features (worst-case per subset)
- NLL maximum, Margin minimum, Confidence 10th percentile
- Rank-based score combination

Usage:
    python best_attack.py [--submit]
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import torchvision.models as models
from tqdm import tqdm

# Configuration
DATASET_PATH = "subsets_dataset.pt"
MODEL_PATH = "classifier.pt"
OUTPUT_PATH = "submission_best.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    """Load the ResNet18 classifier."""
    model = models.resnet18(weights=None)
    model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=3, bias=False)
    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2),
        torch.nn.Linear(model.fc.in_features, 9)
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=False))
    model.to(DEVICE)
    model.eval()
    return model


def load_dataset():
    """Load the subsets dataset."""
    return torch.load(DATASET_PATH, weights_only=False)


def extract_extreme_features(model, dataset):
    """
    Extract extreme-value features for each subset.
    
    Key insight: Focus on WORST-CASE samples per subset, not averages.
    Members should have better worst-case performance (less extreme outliers).
    """
    nll_max = []      # Highest NLL value
    margin_min = []   # Lowest margin value  
    conf_p10 = []     # 10th percentile confidence
    
    with torch.no_grad():
        for i in tqdm(range(1000), desc="Extracting features"):
            subset = dataset[f"subset_{i}"]
            images = subset['images']
            labels = subset['labels']
            
            # Handle channel conversion if needed
            if images.shape[1] == 1:
                images = images.repeat(1, 3, 1, 1)
            
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            
            # Forward pass
            logits = model(images)
            probs = F.softmax(logits, dim=1)
            
            # NLL (Negative Log-Likelihood)
            true_probs = probs.gather(1, labels.view(-1, 1)).squeeze(1)
            nll = -torch.log(true_probs.clamp(min=1e-10))
            
            # Confidence (max probability)
            conf = probs.max(dim=1).values
            
            # Margin (true class logit - max other class logit)
            true_logit = logits.gather(1, labels.view(-1, 1)).squeeze(1)
            mask = torch.ones_like(logits, dtype=torch.bool)
            mask.scatter_(1, labels.view(-1, 1), False)
            other_logits = logits.clone()
            other_logits[~mask] = float('-inf')
            max_other = other_logits.max(dim=1).values
            margin = true_logit - max_other
            
            # Store EXTREME values
            nll_max.append(nll.max().item())
            margin_min.append(margin.min().item())
            conf_p10.append(torch.quantile(conf, 0.1).item())
    
    return {
        'nll_max': np.array(nll_max),
        'margin_min': np.array(margin_min),
        'conf_p10': np.array(conf_p10),
    }


def compute_scores(features):
    """
    Compute membership scores using rank-based combination.
    
    Scoring logic:
    - Lower nll_max = more likely member
    - Higher margin_min = more likely member
    - Higher conf_p10 = more likely member
    """
    ranks = np.zeros(1000)
    
    # nll_max: LOWER is better (member)
    ranks += np.argsort(np.argsort(features['nll_max']))
    
    # margin_min: HIGHER is better (member)
    ranks += np.argsort(np.argsort(-features['margin_min']))
    
    # conf_p10: HIGHER is better (member)
    ranks += np.argsort(np.argsort(-features['conf_p10']))
    
    # Convert to [0, 1] score (lower rank = higher score)
    scores = 1 - (ranks / 3 / 999)
    
    return scores


def create_submission(scores, output_path):
    """Create and validate submission CSV."""
    df = pd.DataFrame({
        'subset_id': range(1000),
        'membership': scores
    })
    
    # Validation
    assert len(df) == 1000
    assert df['subset_id'].min() == 0
    assert df['subset_id'].max() == 999
    assert df['membership'].isna().sum() == 0
    assert np.isfinite(df['membership']).all()
    
    df.to_csv(output_path, index=False)
    print(f"\nSubmission saved: {output_path}")
    print(f"  Score range: [{scores.min():.4f}, {scores.max():.4f}]")
    print(f"  Score mean:  {scores.mean():.4f}")
    
    return df


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--submit', action='store_true', help='Submit to leaderboard')
    args = parser.parse_args()
    
    print("="*60)
    print("Dataset Inference Attack - Best Approach")
    print("="*60)
    print(f"Device: {DEVICE}")
    
    # Load
    print("\n[1/4] Loading model...")
    model = load_model()
    
    print("\n[2/4] Loading dataset...")
    dataset = load_dataset()
    
    # Extract features
    print("\n[3/4] Extracting extreme features...")
    features = extract_extreme_features(model, dataset)
    
    # Compute scores
    print("\n[4/4] Computing membership scores...")
    scores = compute_scores(features)
    
    # Create submission
    create_submission(scores, OUTPUT_PATH)
    
    # Optionally submit
    if args.submit:
        import subprocess
        print("\nSubmitting to leaderboard...")
        result = subprocess.run(
            ['python', 'submit.py', '--submission', OUTPUT_PATH],
            capture_output=True, text=True
        )
        print(result.stdout)
        if result.returncode != 0:
            print(result.stderr)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
