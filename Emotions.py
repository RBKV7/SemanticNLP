import os, json, torch, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from codecarbon import EmissionsTracker
from tensorflow.keras.datasets import imdb as keras_imdb
from typing import Optional


HF_CACHE = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.makedirs(HF_CACHE, exist_ok=True)

# ==============================================================
# Safe loaders with retries
# ==============================================================
def safe_tokenizer_load(model_name: str, cache_dir: Optional[str] = HF_CACHE):
    """
    Load tokenizer robustly: try normal, then force re-download if needed.
    """
    try:
        tok = AutoTokenizer.from_pretrained(model_name, use_fast=True, cache_dir=cache_dir)
        # sanity check: tokenizer has vocab/merges internally set
        return tok
    except Exception as e:
        print(f" Tokenizer load failed for {model_name}: {e}\n→ Retrying with force_download=True...")
        tok = AutoTokenizer.from_pretrained(model_name, use_fast=True, cache_dir=cache_dir, force_download=True)
        return tok

def safe_model_load(model_name: str, num_labels: int = 2, cache_dir: Optional[str] = HF_CACHE):
    """
    Load model robustly: try normal, then force re-download if needed.
    """
    try:
        mdl = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels, cache_dir=cache_dir
        )
        return mdl
    except Exception as e:
        print(f" Model load failed for {model_name}: {e}\n→ Retrying with force_download=True...")
        mdl = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels, cache_dir=cache_dir, force_download=True
        )
        return mdl

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
# Train a single model
# ==============================================================
def train_model(model_name, df, outdir, seed=42, batch_size=16, epochs=3, max_length=256, device: str = "cpu"):
    print(f"\n Training {model_name}...")

    # Split data: 80% train, 10% val, 10% test
    train_df, test_df = train_test_split(df, test_size=0.1, stratify=df["label"], random_state=seed)
    val_df, test_df = train_test_split(test_df, test_size=0.5, stratify=test_df["label"], random_state=seed)

    tokenizer = safe_tokenizer_load(model_name, cache_dir=HF_CACHE)

    def tokenize_fn(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=max_length)

    train_ds = Dataset.from_pandas(train_df).map(tokenize_fn, batched=True)
    val_ds   = Dataset.from_pandas(val_df).map(tokenize_fn, batched=True)
    test_ds  = Dataset.from_pandas(test_df).map(tokenize_fn, batched=True)

    # keep only model inputs + label (avoid __index_level_0__)
    cols_keep = ["input_ids", "attention_mask", "label"]
    for ds in (train_ds, val_ds, test_ds):
        drop_cols = [c for c in ds.column_names if c not in cols_keep]
        if drop_cols:
            ds = ds.remove_columns(drop_cols)
        ds.set_format("torch")
    # reassign after cleaning
    train_ds = train_ds.remove_columns([c for c in train_ds.column_names if c not in cols_keep]).with_format("torch")
    val_ds   = val_ds.remove_columns([c for c in val_ds.column_names if c not in cols_keep]).with_format("torch")
    test_ds  = test_ds.remove_columns([c for c in test_ds.column_names if c not in cols_keep]).with_format("torch")

    model = safe_model_load(model_name, num_labels=2, cache_dir=HF_CACHE)

    model_dir = os.path.join(outdir, model_name.replace("/", "_"))
    os.makedirs(model_dir, exist_ok=True)

    args = TrainingArguments(
        output_dir=model_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
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

    # CO₂ tracking (Lithuania)
    tracker = EmissionsTracker(project_name=model_name, output_dir=outdir, country_iso_code="LT")
    tracker.start()

    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds, compute_metrics=compute_metrics)
    trainer.train()
    emissions = tracker.stop()

    preds = trainer.predict(test_ds)
    y_true = preds.label_ids
    y_pred = np.argmax(preds.predictions, axis=1)
    probs = torch.softmax(torch.tensor(preds.predictions), dim=1)[:, 1].numpy()

    acc = accuracy_score(y_true, y_pred)
    pr, rc, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro")
    print(f" {model_name}: acc={acc:.4f}, f1={f1:.4f}, co₂={emissions:.6f} kg")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(confusion_matrix=cm).plot(values_format="d")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.savefig(os.path.join(model_dir, "cm_test.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # JSON-safe summary
    summary = {
        "name": model_name,
        "acc": float(acc),
        "f1": float(f1),
        "emissions": float(emissions),
        "y_true": np.asarray(y_true).tolist(),
        "y_pred": np.asarray(y_pred).tolist(),
        "probs": np.asarray(probs).tolist()
    }
    with open(os.path.join(model_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary

# ==============================================================
# Soft Voting Ensemble
# ==============================================================
def soft_vote(model_a, model_b):
    # Ensure arrays
    y_true_a = np.asarray(model_a["y_true"])
    y_true_b = np.asarray(model_b["y_true"])
    assert np.array_equal(y_true_a, y_true_b), "y_true mismatch between models"
    y_true = y_true_a

    p_a = np.asarray(model_a["probs"])
    p_b = np.asarray(model_b["probs"])

    w = np.array([model_a["f1"], model_b["f1"]], dtype=float)
    w = w / w.sum()
    p = w[0] * p_a + w[1] * p_b
    y_pred = (p >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    pr, rc, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro")
    print(f"\n Soft-vote ensemble (RoBERTa + DistilBERT): acc={acc:.4f}, f1={f1:.4f}")
    return {"acc": float(acc), "f1": float(f1)}

# ==============================================================
# Demo on custom texts
# ==============================================================
def demo_texts_return(model_name, ckpt_dir, texts, device="cpu"):
    print(f"\n[DEMO] {model_name} on sample sentences")

    subdirs = [d for d in os.listdir(ckpt_dir) if d.startswith("checkpoint-")]
    if subdirs and not any(f.endswith((".bin", ".safetensors")) for f in os.listdir(ckpt_dir)):
        latest = sorted(subdirs, key=lambda x: int(x.split("-")[-1]))[-1]
        ckpt_dir = os.path.join(ckpt_dir, latest)

    base_model = "roberta-base" if "roberta" in model_name else "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt_dir).to(device).eval()

    enc = tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**enc)
        probs = torch.softmax(out.logits, dim=1)[:, 1].cpu().numpy()
        preds = (probs >= 0.5).astype(int)

    for t, p, pr in zip(texts, preds, probs):
        label = "pos" if p else "neg"
        print(f"{label:>3} ({pr:.3f}) → {t[:120]}")

    return preds, probs

# ==============================================================
# MAIN
# ==============================================================
# ==============================================================
# MAIN (Load pretrained models + evaluate custom sentences)
# ==============================================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    outdir = "outputs_transformer_comparison"
    os.makedirs(outdir, exist_ok=True)

    print(f"[DEVICE] {device}")
    print("[MODELS] Loading pretrained checkpoints for demo...")

    # Paths to your pretrained checkpoints
    pretrained_dirs = {
        "roberta-base": os.path.join(outdir, "roberta-base", "checkpoint-1760"),
        "distilbert-base-uncased": os.path.join(outdir, "distilbert-base-uncased", "checkpoint-1760"),
    }

    # List of test sentences
    samples = [
        "I absolutely loved this movie. The plot was engaging and the acting was brilliant.",
        "Terrible film. Boring, predictable and a complete waste of time.",
        "Not bad, but the pacing was off and the ending didn't land.",
        "This was one of the best performances I've ever seen from the lead actor!",
        "The movie started strong but completely lost focus halfway through.",
        "Mediocre at best — nothing new, just another cliché Hollywood remake.",
        "What a masterpiece! Beautiful cinematography and an emotional storyline.",
        "I fell asleep after twenty minutes. It's painfully slow and confusing.",
        "The soundtrack was amazing, but the script felt very weak.",
        "Surprisingly good! I expected a disaster, but it turned out to be very enjoyable.",
        "Disappointing. The trailer was much better than the actual film.",
        "A deeply moving story with relatable characters and stunning visuals.",
        "The jokes were terrible and the dialogue was cringe-worthy.",
        "Neutral feelings — it wasn’t bad, but I wouldn’t watch it again.",
        "Wow, I didn’t expect to cry at the end. What an emotional journey!",
        "Just awful. I can’t believe I wasted my money on this.",
        "An instant classic! I’ll be recommending this to all my friends.",
        "It had potential, but the execution was lacking in many areas.",
        "A fun ride from start to finish. Great for a casual movie night.",
        "I regret watching this. The plot made no sense and the characters were flat."
    ]

    # Evaluate each pretrained model
    all_preds = []
    for model_name, ckpt_dir in pretrained_dirs.items():
        print(f"\n--- Evaluating {model_name} ---")
        preds, probs = demo_texts_return(model_name, ckpt_dir, samples, device)
        all_preds.append(probs)

    # Soft-vote ensemble across models
    if len(all_preds) > 1:
        ensemble_probs = np.mean(np.vstack(all_preds), axis=0)
        ensemble_preds = (ensemble_probs >= 0.5).astype(int)
        print("\n[SOFT-VOTE ENSEMBLE RESULTS]")
        for text, p, pr in zip(samples, ensemble_preds, ensemble_probs):
            label = "pos" if p else "neg"
            print(f"{label:>3} ({pr:.3f}) → {text[:120]}")

    print("\n Evaluation complete.")

if __name__ == "__main__":
    main()