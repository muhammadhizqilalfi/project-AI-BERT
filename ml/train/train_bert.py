import os
import torch
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import (
    TrainingArguments,
    Trainer,
    AutoTokenizer,
    AutoModelForSequenceClassification
)
import numpy as np
import evaluate

# ==========================================================
# PATH HANDLING
# ==========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))      
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

DATA_PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")
MODEL_OUTPUT_DIR = os.path.join(ROOT_DIR, "models", "sentiment_indobert")

train_csv = os.path.join(DATA_PROCESSED_DIR, "train.csv")
test_csv  = os.path.join(DATA_PROCESSED_DIR, "test.csv")

os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

print(">>> Train CSV :", train_csv)
print(">>> Test  CSV :", test_csv)
print(">>> Output dir:", MODEL_OUTPUT_DIR)

# ==========================================================
# TRAINING CONFIG
# ==========================================================
MODEL_NAME = "indobenchmark/indobert-base-p1"
MAX_LEN = 128
BATCH = 16
EPOCHS = 3

device = "cuda" if torch.cuda.is_available() else "cpu"
print(">>> Running on:", device)


# ==========================================================
# LOAD DATASET (FIXED)
# ==========================================================
def load_dataset():
    train = pd.read_csv(train_csv)
    test = pd.read_csv(test_csv)

    print(">>> Train columns:", list(train.columns))

    # ======================================================
    # AUTO DETECT LABEL COLUMN
    # ======================================================
    possible_label_cols = ["sentiment", "Sentiment", "label", "Label"]
    label_col = next((c for c in possible_label_cols if c in train.columns), None)

    if label_col is None:
        raise ValueError("Kolom label tidak ditemukan di train.csv")

    text_col = "clean_text"

    print(f">>> Text column  : {text_col}")
    print(f">>> Label column : {label_col}")

    # ======================================================
    # NORMALISASI LABEL KE 0–2
    # ======================================================
    unique_labels = sorted(train[label_col].unique())
    print(">>> Unique labels before:", unique_labels)

    # {-1, 0, 1} → {0, 1, 2}
    if set(unique_labels) == {-1, 0, 1}:
        mapping = {-1: 0, 0: 1, 1: 2}
        train[label_col] = train[label_col].map(mapping)
        test[label_col] = test[label_col].map(mapping)
        print(">>> Label mapped {-1,0,1} → {0,1,2}")

    # {1, 2, 3} → {0, 1, 2}
    elif set(unique_labels) == {1, 2, 3}:
        mapping = {1: 0, 2: 1, 3: 2}
        train[label_col] = train[label_col].map(mapping)
        test[label_col] = test[label_col].map(mapping)
        print(">>> Label mapped {1,2,3} → {0,1,2}")

    print(">>> Unique labels after:", sorted(train[label_col].unique()))

    # ======================================================
    # CONVERT TO HUGGINGFACE DATASET
    # ======================================================
    train_ds = Dataset.from_pandas(
        train[[text_col, label_col]].rename(
            columns={text_col: "text", label_col: "label"}
        )
    )

    test_ds = Dataset.from_pandas(
        test[[text_col, label_col]].rename(
            columns={text_col: "text", label_col: "label"}
        )
    )

    return DatasetDict({"train": train_ds, "test": test_ds})


# ==========================================================
# TOKENIZER
# ==========================================================
def tokenize(batch, tokenizer):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN
    )


# ==========================================================
# METRICS
# ==========================================================
acc_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    acc = acc_metric.compute(predictions=preds, references=labels)["accuracy"]
    f1_macro = f1_metric.compute(predictions=preds, references=labels, average="macro")["f1"]

    return {
        "accuracy": acc,
        "f1_macro": f1_macro
    }


# ==========================================================
# PREDICT FUNCTION
# ==========================================================
def predict_text(model, tokenizer, text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LEN
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
        pred = torch.argmax(logits, dim=-1).item()

    label_map = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}
    return label_map[pred]


# ==========================================================
# MAIN
# ==========================================================
def main():
    ds = load_dataset()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    ds = ds.map(lambda x: tokenize(x, tokenizer), batched=True)

    keep = ["input_ids", "attention_mask", "label"]
    if "token_type_ids" in ds["train"].column_names:
        keep.append("token_type_ids")

    ds = ds.remove_columns([c for c in ds["train"].column_names if c not in keep])
    ds.set_format("torch")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3
    ).to(device)

    args = TrainingArguments(
        output_dir=MODEL_OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH,
        per_device_eval_batch_size=BATCH,
        learning_rate=2e-5,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),

        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=100,

        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        tokenizer=tokenizer,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        compute_metrics=compute_metrics
    )

    trainer.train()

    print("\n====================")
    print(" FINAL EVALUATION")
    print("====================")
    print(trainer.evaluate())

    trainer.save_model(MODEL_OUTPUT_DIR)
    tokenizer.save_pretrained(MODEL_OUTPUT_DIR)

    print("\n=========== SAMPLE PREDICTION ===========")
    samples = [
        "pemerintah menyusahkan rakyat",
        "saya senang dengan kebijakan baru",
        "kebijakan ini sangat buruk",
        "bagus sekali pelayanannya"
    ]

    for s in samples:
        print(s, " => ", predict_text(model, tokenizer, s))

    print("\n>>> Model saved in:", MODEL_OUTPUT_DIR)


if __name__ == "__main__":
    main()
