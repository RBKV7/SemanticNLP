import os, json, torch, inspect
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from codecarbon import EmissionsTracker
from tensorflow.keras.datasets import imdb as keras_imdb

# ==============================================================
# Load IMDB dataset
# ==============================================================
def _decode_imdb_review(encoded, reverse_index):
    """Decode IMDB integer sequences into plain text."""
    return " ".join([reverse_index.get(i - 3, "?") for i in encoded])

def load_imdb_dataframe(max_words=20000, fraction=0.8, seed=42):
    """Load IMDB reviews from Keras and return a DataFrame."""
    num_words = None if (max_words <= 0) else max_words
    (Xtr, ytr), (Xte, yte) = keras_imdb.load_data(num_words=num_words)
    idx = keras_imdb.get_word_index()
    rev = {v: k for k, v in idx.items()}
    texts = [_decode_imdb_review(seq, rev) for seq in list(Xtr) + list(Xte)]
    labels = list(ytr) + list(yte)
    df = pd.DataFrame({"text": texts, "label": labels})
    if 0 < fraction < 1.0:
        df = df.sample(frac=fraction, random_state=seed).reset_index(drop=True)
    return df

# ==============================================================
# Build TrainingArguments (auto-detects correct keyword)
# ==============================================================
def make_training_args(model_dir, outdir, batch_size, epochs, seed, lr=1.5e-5):
    """Creates TrainingArguments compatible with both old and new Transformers versions."""
    sig = inspect.signature(TrainingArguments)
    kwargs = dict(
        output_dir=model_dir,
        save_strategy="epoch",
        learning_rate=lr,
        weight_decay=0.01,
        gradient_accumulation_steps=2,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_dir=os.path.join(outdir, "logs", os.path.basename(model_dir)),
        logging_steps=100,
        evaluation_strategy="epoch" if "evaluation_strategy" in sig.parameters else None,
        seed=seed,
        report_to="none"
    )
    # Compatibility for new versions (>=4.56)
    if "evaluation_strategy" not in sig.parameters:
        kwargs["eval_strategy"] = "epoch"
        del kwargs["evaluation_strategy"]
    return TrainingArguments(**{k: v for k, v in kwargs.items() if v is not None})

# ==============================================================
# Train + evaluate one model
# ==============================================================
def run_transformer(model_name, df, outdir, seed=42, batch_size=16, epochs=6, max_length=256):
    print(f"\n Training {model_name}...")

    # Split data
    train_df, test_df = train_test_split(df, test_size=0.1, stratify=df["label"], random_state=seed)
    val_df, test_df = train_test_split(test_df, test_size=0.5, stratify=test_df["label"], random_state=seed)

    # Tokenization
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    def tokenize_fn(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=max_length)

    train_ds = Dataset.from_pandas(train_df).map(tokenize_fn, batched=True)
    val_ds   = Dataset.from_pandas(val_df).map(tokenize_fn, batched=True)
    test_ds  = Dataset.from_pandas(test_df).map(tokenize_fn, batched=True)
    for ds in [train_ds, val_ds, test_ds]:
        ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    # Model setup
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model_dir = os.path.join(outdir, model_name.replace("/", "_"))
    os.makedirs(model_dir, exist_ok=True)

    # Training args
    args_hf = make_training_args(model_dir, outdir, batch_size, epochs, seed)

    # Metrics
    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        preds = np.argmax(preds, axis=1)
        acc = accuracy_score(labels, preds)
        pr, rc, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
        return {"accuracy": acc, "precision": pr, "recall": rc, "f1": f1}

    # CodeCarbon tracking (Lithuania)
    tracker = EmissionsTracker(
        project_name=f"{model_name}_experiment",
        output_dir=outdir
    )
    tracker.start()

    trainer = Trainer(
        model=model,
        args=args_hf,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics
    )
    trainer.train()

    emissions = tracker.stop()

    # Evaluate on test set
    preds = trainer.predict(test_ds)
    y_true = preds.label_ids
    y_pred = np.argmax(preds.predictions, axis=1)
    acc = accuracy_score(y_true, y_pred)
    pr, rc, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro")

    # Save per-model outputs
    pd.DataFrame(classification_report(y_true, y_pred, output_dict=True)).transpose().to_csv(
        os.path.join(model_dir, "report.csv")
    )

    cm = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(confusion_matrix=cm).plot(values_format="d")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.savefig(os.path.join(model_dir, "confusion_matrix.png"), dpi=150, bbox_inches="tight")
    plt.close()

    summary = {
        "model": model_name,
        "accuracy": float(acc),
        "precision": float(pr),
        "recall": float(rc),
        "f1": float(f1),
        "emissions_kg": float(emissions),
    }
    with open(os.path.join(model_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f" {model_name}: acc={acc:.4f}, f1={f1:.4f}, emissions={emissions:.6f} kgCO₂e")
    return summary

# ==============================================================
# MAIN
# ==============================================================
def main():
    torch.manual_seed(42)
    outdir = "outputs_transformer_comparison"
    os.makedirs(outdir, exist_ok=True)

    print("[DATA] Loading IMDB dataset (25%)...")
    df = load_imdb_dataframe(fraction=0.25, seed=42)
    df = df.astype({"text": str, "label": int})

    models = [
        "bert-base-uncased",
        "roberta-base",
        "xlnet-base-cased",
        "distilbert-base-uncased",
        "albert-base-v2"
    ]

    summaries = [run_transformer(m, df, outdir) for m in models]

    df_summary = pd.DataFrame(summaries)
    df_summary.to_csv(os.path.join(outdir, "all_models_summary.csv"), index=False)

    # Plot
    plt.figure(figsize=(8, 5))
    plt.scatter(df_summary["emissions_kg"], df_summary["accuracy"], s=120, c="darkorange")
    for _, row in df_summary.iterrows():
        plt.text(row["emissions_kg"] + 0.00005, row["accuracy"], row["model"].split("-")[0], fontsize=9)
    plt.xlabel("Emissions (kg CO₂e)")
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy vs Energy Emissions (Lithuania)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "comparison_plot.png"), dpi=150)
    plt.close()

    print("\n Final comparison:")
    print(df_summary)
    print(f"\n Results and plots saved in: {outdir}")

if __name__ == "__main__":
    main()