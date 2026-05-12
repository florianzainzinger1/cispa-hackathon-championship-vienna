"""
Test-Time Augmentation (TTA) for ModernBERT-Large
Runs predictions on multiple augmented versions of each text.
Averages the predictions for better robustness.
"""
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import os
import random

# --- CONFIG ---
MODEL_PATH = "/p/scratch/training2562/zainzinger1/model_modernbert_large_trained"
TEST_FILE = "participants_test.csv"
OUTPUT_FILE = "submission_tta.csv"
BATCH_SIZE = 16
NUM_AUGMENTATIONS = 3  # Original + 2 augmentations

# Augmentation functions
def augment_whitespace(text):
    """Add random extra spaces"""
    words = text.split()
    if len(words) > 10:
        # Add double spaces randomly
        for _ in range(random.randint(1, 3)):
            idx = random.randint(0, len(words)-2)
            words[idx] = words[idx] + "  "
    return " ".join(words)

def augment_case(text):
    """Randomly change case of some words"""
    words = text.split()
    if len(words) > 5:
        for _ in range(min(3, len(words)//10)):
            idx = random.randint(0, len(words)-1)
            if random.random() > 0.5:
                words[idx] = words[idx].upper()
            else:
                words[idx] = words[idx].lower()
    return " ".join(words)

def augment_text(text, aug_type):
    """Apply augmentation based on type"""
    if aug_type == 0:
        return text  # Original
    elif aug_type == 1:
        return augment_whitespace(text)
    elif aug_type == 2:
        return augment_case(text)
    return text

def predict_with_tta(model, tokenizer, texts, labels, device):
    """Run predictions with TTA"""
    all_probs = []
    
    for aug_idx in range(NUM_AUGMENTATIONS):
        print(f"Augmentation {aug_idx+1}/{NUM_AUGMENTATIONS}...")
        
        # Augment texts
        aug_texts = [augment_text(t, aug_idx) for t in texts]
        
        # Batch predictions
        batch_probs = []
        for i in tqdm(range(0, len(aug_texts), BATCH_SIZE), desc=f"Aug {aug_idx+1}"):
            batch = aug_texts[i:i+BATCH_SIZE]
            
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(device)
            
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                batch_probs.append(probs.cpu().numpy())
        
        # Concatenate batch results
        aug_probs = np.vstack(batch_probs)
        all_probs.append(aug_probs)
    
    # Average across augmentations
    avg_probs = np.mean(all_probs, axis=0)
    
    # Get predictions
    predictions = [labels[p.argmax()] for p in avg_probs]
    
    return predictions, avg_probs

def main():
    print("=== TEST-TIME AUGMENTATION ===")
    print(f"Using {NUM_AUGMENTATIONS} augmentations")
    
    # Load model
    print(f"Loading model from {MODEL_PATH}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(device)
    model.eval()
    
    # Get labels
    labels = sorted([
        'CohereLabs/c4ai-command-r7b-12-2024',
        'Qwen/Qwen1.5-14B-Chat',
        'deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct',
        'gpt-4.1',
        'gpt-4.1-nano',
        'gpt-5-chat-latest',
        'human',
        'ibm-granite/granite-3.0-8b-instruct',
        'marin-community/marin-8b-instruct',
        'meta-llama/Llama-3.1-8B-Instruct',
        'microsoft/Phi-3-medium-128k-instruct',
        'mistralai/Mistral-7B-Instruct-v0.3',
        'outlier'
    ])
    
    # Load test data
    print(f"Loading test data from {TEST_FILE}...")
    test_df = pd.read_csv(TEST_FILE)
    test_df["response"] = test_df["response"].fillna("").astype(str)
    texts = test_df["response"].tolist()
    
    print(f"Test samples: {len(texts)}")
    
    # Run TTA
    predictions, probs = predict_with_tta(model, tokenizer, texts, labels, device)
    
    # Save submission
    submission = pd.DataFrame({
        "uuid": test_df["uuid"],
        "model": predictions
    })
    submission.to_csv(OUTPUT_FILE, index=False)
    
    print(f"\n=== Saved: {OUTPUT_FILE} ===")
    print(submission["model"].value_counts().head())

if __name__ == "__main__":
    # Set seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    main()
