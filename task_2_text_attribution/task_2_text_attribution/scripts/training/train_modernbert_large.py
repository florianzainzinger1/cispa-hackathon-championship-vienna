import pandas as pd
import numpy as np
import os
import torch

# Fix für torch.compile Probleme
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.disable = True
os.environ["TORCH_COMPILE_DISABLE"] = "1"

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding,
    EarlyStoppingCallback
)
from datasets import Dataset

# --- CONFIG: ModernBERT-LARGE (OFFLINE) ---
SCRATCH_DIR = "/p/scratch/training2562/zainzinger1"
MODEL_NAME = os.path.join(SCRATCH_DIR, "models/modernbert-large")
OUTPUT_DIR = os.path.join(SCRATCH_DIR, "model_modernbert_large_trained")

MAX_LEN = 512 
BATCH_SIZE = 16  # Kleiner wegen größerem Modell
EPOCHS = 5
LR = 2e-5  # Niedriger für großes Modell

# --- OVERFITTING CONTROL ---
LABEL_SMOOTHING = 0.1
WARMUP_RATIO = 0.1
EARLY_STOPPING_PATIENCE = 2

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    return {"f1_macro": f1_score(labels, preds, average="macro")}

def main():
    print(f"=== STARTE TRAINING: ModernBERT-LARGE ===")
    print(f"Modell: {MODEL_NAME}")
    print(f"Output: {OUTPUT_DIR}")
    
    if not os.path.exists("participants_train.csv"):
        raise FileNotFoundError("participants_train.csv nicht gefunden!")

    df = pd.read_csv("participants_train.csv")
    
    # CLEANING
    df = df.dropna(subset=["response"])
    df["response"] = df["response"].astype(str)

    labels = sorted(df['model'].unique().tolist())
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}
    df['label'] = df['model'].map(label2id)
    
    print(f"Anzahl Klassen: {len(labels)}")
    
    train_df, val_df = train_test_split(df, test_size=0.1, stratify=df['label'], random_state=42)
    ds_train = Dataset.from_pandas(train_df)
    ds_val = Dataset.from_pandas(val_df)
    
    print("Lade Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    def token_fn(examples):
        return tokenizer(examples["response"], truncation=True, max_length=MAX_LEN, padding="max_length")
        
    print("Tokenisiere...")
    ds_train = ds_train.map(token_fn, batched=True, load_from_cache_file=False)
    ds_val = ds_val.map(token_fn, batched=True, load_from_cache_file=False)

    print("Lade Modell...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=len(labels), 
        id2label=id2label, 
        label2id=label2id,
        trust_remote_code=True,
        ignore_mismatched_sizes=True
    )
    
    model.config.pad_token_id = tokenizer.pad_token_id
    
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=LR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE*2,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        fp16=True,
        report_to="none",
        save_total_limit=1,
        label_smoothing_factor=LABEL_SMOOTHING,
        warmup_ratio=WARMUP_RATIO,
        gradient_accumulation_steps=2,  # Effektive batch size = 32
        logging_steps=100
    )
    
    trainer = Trainer(
        model=model, args=args,
        train_dataset=ds_train, eval_dataset=ds_val,
        tokenizer=tokenizer, data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)]
    )
    
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"=== FERTIG! Modell gespeichert in {OUTPUT_DIR} ===")

if __name__ == "__main__":
    main()
