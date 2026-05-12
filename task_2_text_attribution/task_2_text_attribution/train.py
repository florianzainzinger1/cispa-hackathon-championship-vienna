import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from datasets import Dataset
import os

# --- KONFIGURATION ---
# Wir starten mit "base". Das passt in den Speicher und lernt schnell.
MODEL_NAME = "microsoft/deberta-v3-base" 
MAX_LENGTH = 512   # DeBERTa sieht max 512 Tokens (Wörter)
BATCH_SIZE = 8     # 8 pro GPU-Schritt ist sicher für A100
EPOCHS = 3         # 3 Durchläufe durch alle Daten reichen meist
LR = 2e-5          # Lernrate (Standard für Fine-Tuning)

def main():
    print("=== START TRAINING ===")
    
    # 1. DATEN LADEN
    print("Lade CSV...")
    if not os.path.exists("participants_train.csv"):
        raise FileNotFoundError("FEHLER: participants_train.csv nicht gefunden!")
        
    df = pd.read_csv("participants_train.csv")
    
    # Wir müssen die Text-Klassen (z.B. "gpt-4") in Zahlen (0, 1, 2...) umwandeln
    labels_list = sorted(list(df['model'].unique()))
    label2id = {label: i for i, label in enumerate(labels_list)}
    id2label = {i: label for label, i in label2id.items()}
    
    print(f"Erkannte Klassen: {len(labels_list)}")
    
    # Neue Spalte 'label' mit den Zahlen erstellen
    df['label'] = df['model'].map(label2id)

    # 10% der Daten zum Testen (Validieren) beiseite legen
    train_df, val_df = train_test_split(df, test_size=0.1, stratify=df['label'], random_state=42)
    
    # Umwandeln in das Format, das HuggingFace mag
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    # 2. TOKENIZER (Der Übersetzer Text -> Zahlen)
    print("Lade Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def preprocess_function(examples):
        return tokenizer(
            examples["response"], 
            truncation=True, 
            max_length=MAX_LENGTH
        )

    print("Tokenisiere Daten (das dauert kurz)...")
    tokenized_train = train_dataset.map(preprocess_function, batched=True)
    tokenized_val = val_dataset.map(preprocess_function, batched=True)

    # 3. MODELL LADEN
    print(f"Lade Modell: {MODEL_NAME}...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(labels_list),
        id2label=id2label,
        label2id=label2id
    )

    # 4. METRIK (Wie gut sind wir?)
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        preds = np.argmax(predictions, axis=1)
        # Macro F1 ist die Metrik aus der Aufgabenstellung
        f1 = f1_score(labels, preds, average="macro")
        return {"f1_macro": f1}

    # 5. TRAINING SETUP
    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=LR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        evaluation_strategy="epoch", # Nach jeder Epoche testen
        save_strategy="epoch",       # Nach jeder Epoche speichern
        load_best_model_at_end=True, # Am Ende das beste Modell behalten
        metric_for_best_model="f1_macro",
        fp16=True,                   # Turbo-Modus auf A100 GPUs
        logging_steps=50,
        report_to="none"             # Kein WandB Login nötig
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # 6. FEUER FREI!
    print("Starte Training...")
    trainer.train()

    # 7. SPEICHERN
    print("Speichere das fertige Modell in ./my_best_model")
    trainer.save_model("./my_best_model")
    tokenizer.save_pretrained("./my_best_model")
    
    print("=== FERTIG! ===")

if __name__ == "__main__":
    main()