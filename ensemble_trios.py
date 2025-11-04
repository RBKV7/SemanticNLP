import os, json, torch, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, classification_report
)
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from codecarbon import EmissionsTracker
from tensorflow.keras.datasets import imdb as keras_imdb

# ==============================================================
# Load IMDB dataset
# ==============================================================
def _decode_imdb_review(encoded, reverse_index):
    return " ".join([reverse_index.get(i - 3, "?") for i in encoded])

def load_imdb_dataframe(max_words=20000, fraction=0.25, seed=42):
    (Xtr, ytr), (Xte, yte) = keras_imdb.load_data(num_words=max_words)
    idx = keras_imdb.get_word_index()
    rev = {v: k for k, v in idx.items()}
    texts = [_decode_imdb_review(seq, rev) for seq in list(Xtr) + list(Xte)]
    labels = list(ytr) + list(yte)
    df = pd.DataFrame({"text": texts, "label": labels})
    if 0 < fraction < 1.0:
        df = df.sample(frac=fraction, random_state=seed).reset_index(drop=True)
    return df

# ==============================================================
# Train one model
# ==============================================================
def train_model(model_name, df, outdir, seed=42, batch_size=16, epochs=3, max_length=256):
    print(f"\n🚀 Training {model_name}...")

    # Split data
    train_df, test_df = train_test_split(df, test_size=0.1, stratify=df["label"], random_state=seed)
    val_df, test_df = train_test_split(test_df, test_size=0.5, stratify=test_df["label"], random_state=seed)

    # Tokenize
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    def tokenize_fn(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=max_length)

    train_ds = Dataset.from_pandas(train_df).map(tokenize_fn, batched=True)
    val_ds   = Dataset.from_pandas(val_df).map(tokenize_fn, batched=True)
    test_ds  = Dataset.from_pandas(test_df).map(tokenize_fn, batched=True)
    for ds in [train_ds, val_ds, test_ds]:
        ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    # Model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    model_dir = os.path.join(outdir, model_name.replace("/", "_"))
    os.makedirs(model_dir, exist_ok=True)

    args = TrainingArguments(
        output_dir=model_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=1.5e-5,
        weight_decay=0.01,
        gradient_accumulation_steps=2,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_dir=os.path.join(outdir, "logs", os.path.basename(model_dir)),
        logging_steps=100,
        seed=seed,
        report_to="none"
    )

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        preds = np.argmax(preds, axis=1)
        acc = accuracy_score(labels, preds)
        pr, rc, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
        return {"accuracy": acc, "precision": pr, "recall": rc, "f1": f1}

    tracker = EmissionsTracker(
        project_name=model_name, output_dir=outdir, country_iso_code="LT"
    )
    tracker.start()

    trainer = Trainer(model=model, args=args, train_dataset=train_ds,
                      eval_dataset=val_ds, compute_metrics=compute_metrics)
    trainer.train()
    emissions = tracker.stop()

    preds = trainer.predict(test_ds)
    y_true = preds.label_ids
    y_pred = np.argmax(preds.predictions, axis=1)
    probs = torch.nn.functional.softmax(torch.tensor(preds.predictions), dim=1)[:, 1].numpy()

    acc = accuracy_score(y_true, y_pred)
    pr, rc, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro")

    print(f"✅ {model_name}: acc={acc:.4f}, f1={f1:.4f}, co₂={emissions:.5f} kg")

    return {
        "name": model_name,
        "acc": acc,
        "f1": f1,
        "emissions": emissions,
        "y_true": y_true,
        "y_pred": y_pred,
        "probs": probs
    }

# ==============================================================
# Ensemble Voting
# ==============================================================
def ensemble_results(models, name, method="weighted"):
    assert all(np.array_equal(models[0]["y_true"], m["y_true"]) for m in models)
    y_true = models[0]["y_true"]
    probs = np.stack([m["probs"] for m in models], axis=1)
    emissions = sum(m["emissions"] for m in models)

    if method == "majority":
        preds = (np.mean([m["y_pred"] for m in models], axis=0) >= 0.5).astype(int)
    elif method == "weighted":
        weights = np.array([m["f1"] for m in models])
        preds = (np.average(probs, axis=1, weights=weights) >= 0.5).astype(int)
    else:
        raise ValueError("method must be 'majority' or 'weighted'")

    acc = accuracy_score(y_true, preds)
    pr, rc, f1, _ = precision_recall_fscore_support(y_true, preds, average="macro")
    print(f"🌿 Ensemble {name} ({method}): acc={acc:.4f}, f1={f1:.4f}, co₂={emissions:.5f} kg")

    return {"ensemble": name, "method": method, "acc": acc, "f1": f1,
            "precision": pr, "recall": rc, "emissions": emissions,
            "efficiency": acc / emissions if emissions > 0 else 0}

# ==============================================================
# MAIN
# ==============================================================
def main():
    torch.manual_seed(42)
    outdir = "outputs_trios"
    os.makedirs(outdir, exist_ok=True)

    print("[DATA] Loading IMDB dataset (25%)...")
    df = load_imdb_dataframe(fraction=0.25, seed=42)
    df = df.astype({"text": str, "label": int})

    max_models = [
        "bert-base-uncased",
        "roberta-base",
        "xlnet-base-cased"
    ]
    green_models = [
        "albert-base-v2",
        "distilbert-base-uncased",
        "roberta-base"
    ]

    results = {}

    # Train Max Accuracy Trio
    print("\n Training Max Accuracy Trio models...")
    results["max"] = [train_model(m, df, outdir) for m in max_models]

    # Train Green Trio
    print("\n Training Green Trio models...")
    results["green"] = [train_model(m, df, outdir) for m in green_models]

    # --- Ensembles ---
    ensembles = []
    ensembles.append(ensemble_results(results["max"], "Max Accuracy Trio", method="majority"))
    ensembles.append(ensemble_results(results["max"], "Max Accuracy Trio", method="weighted"))
    ensembles.append(ensemble_results(results["green"], "Green Trio", method="majority"))
    ensembles.append(ensemble_results(results["green"], "Green Trio", method="weighted"))

    df_ens = pd.DataFrame(ensembles)
    df_ens.to_csv(os.path.join(outdir, "ensemble_trio_results.csv"), index=False)

    # --- Plot ---
    plt.figure(figsize=(8, 5))
    plt.scatter(df_ens["emissions"], df_ens["acc"], s=120, c=["blue" if "Max" in e else "green" for e in df_ens["ensemble"]])
    for _, row in df_ens.iterrows():
        plt.text(row["emissions"] + 0.0001, row["acc"], f"{row['ensemble']} ({row['method']})", fontsize=8)
    plt.xlabel("CO₂ emissions (kg)")
    plt.ylabel("Accuracy")
    plt.title("Transformer Ensemble Comparison (Lithuania)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "ensemble_trio_plot.png"), dpi=150)
    plt.close()

    print("\n Results saved to:", os.path.join(outdir, "ensemble_trio_results.csv"))
    print(df_ens)

if __name__ == "__main__":
    main()