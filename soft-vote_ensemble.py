import os, json, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset
from tensorflow.keras.datasets import imdb as keras_imdb

# ---------- Config ----------
OUTDIR = "outputs_transformer_comparison"
EMISSIONS_FILE = os.path.join(OUTDIR, "emissions_v1.csv")
MODELS = ["roberta-base", "distilbert-base-uncased", "albert-base-v2"]
FRACTION = 0.25
SEED = 42
MAX_LEN = 256
BATCH_SIZE = 16


# ---------- Load emissions ----------
def load_emissions_data(filepath):
    if not os.path.exists(filepath):
        print(f"️ Emissions file not found: {filepath}")
        return {}
    df = pd.read_csv(filepath)
    if "project_name" not in df.columns or "emissions" not in df.columns:
        raise ValueError(" CSV must contain 'project_name' and 'emissions' columns.")

    df["model"] = df["project_name"].str.replace("_experiment", "", regex=False).str.lower()
    emissions_map = {row["model"]: float(row["emissions"]) for _, row in df.iterrows()}

    print(f" Loaded emissions data for {len(emissions_map)} models from {filepath}")
    for k, v in emissions_map.items():
        print(f"   {k:30s} → {v:.6f} kg CO₂")
    return emissions_map


# ---------- IMDB load ----------
def _decode_imdb_review(encoded, reverse_index):
    return " ".join([reverse_index.get(i - 3, "?") for i in encoded])

def load_imdb_dataframe(max_words=20000, fraction=FRACTION, seed=SEED):
    (Xtr, ytr), (Xte, yte) = keras_imdb.load_data(num_words=max_words)
    idx = keras_imdb.get_word_index()
    rev = {v: k for k, v in idx.items()}
    texts = [_decode_imdb_review(seq, rev) for seq in list(Xtr) + list(Xte)]
    labels = list(ytr) + list(yte)
    df = pd.DataFrame({"text": texts, "label": labels})
    if 0 < fraction < 1.0:
        df = df.sample(frac=fraction, random_state=seed).reset_index(drop=True)
    return df

def make_splits(df, seed=SEED):
    train_df, test_df = train_test_split(df, test_size=0.1, stratify=df["label"], random_state=seed)
    val_df, test_df = train_test_split(test_df, test_size=0.5, stratify=test_df["label"], random_state=seed)
    return train_df, val_df, test_df


# ---------- Tokenize ----------
def tokenize_dataset(df, tokenizer, max_length=MAX_LEN):
    ds = Dataset.from_pandas(df)
    def tok_fn(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=max_length)
    ds = ds.map(tok_fn, batched=True)
    ds = ds.remove_columns([c for c in ds.column_names if c not in ["input_ids", "attention_mask", "label"]])
    ds.set_format("torch")
    return ds


# ---------- Predict helpers ----------
@torch.no_grad()
def predict_logits(model, ds, batch_size=BATCH_SIZE, device="cpu"):
    model.eval().to(device)
    logits_all, labels_all = [], []
    for i in range(0, len(ds), batch_size):
        batch = ds[i : i+batch_size]
        inputs = {k: v.to(device) for k, v in batch.items() if k in ["input_ids", "attention_mask"]}
        outputs = model(**inputs)
        logits_all.append(outputs.logits.cpu())
        labels_all.append(batch["label"].cpu())
    logits = torch.cat(logits_all, dim=0).numpy()
    labels = torch.cat(labels_all, dim=0).numpy()
    return logits, labels


# ---------- Evaluation ----------
def evaluate_model(ckpt_dir, df_data, emissions_data, device="cpu", split_name="test"):
    base_model = None
    for m in MODELS:
        if m.replace("/", "_") in ckpt_dir:
            base_model = m
            break
    if base_model is None:
        base_model = "roberta-base"

    print(f"\n===> Evaluating {base_model} ({split_name})")
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt_dir)
    ds_data = tokenize_dataset(df_data, tokenizer, max_length=MAX_LEN)

    logits, y_true = predict_logits(model, ds_data, device=device)
    probs = torch.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()
    y_pred = np.argmax(logits, axis=1)

    acc = accuracy_score(y_true, y_pred)
    pr, rc, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)

    emission_value = emissions_data.get(base_model.lower(), 0.0)
    print(f"acc={acc:.4f} | f1={f1:.4f} | co₂(from file)={emission_value:.6f} kg")

    return {
        "name": base_model,
        "acc": acc,
        "f1": f1,
        "emissions": emission_value,
        "y_true": y_true,
        "y_pred": y_pred,
        "probs": probs
    }


# ---------- Meta-Ensemble (XGBoost + Confidence + Entropy) ----------
def stacked_ensemble_xgb(model_results_val, model_results_test):
    print("\n Training XGBoost meta-ensemble with confidence and entropy features...")

    y_val = model_results_val[0]["y_true"]
    y_test = model_results_test[0]["y_true"]

    # Build feature matrix from model probabilities
    def make_features(model_results):
        probs = np.vstack([m["probs"] for m in model_results]).T
        conf = np.abs(probs - 0.5)
        entropy = -(probs * np.log(probs + 1e-9) + (1 - probs) * np.log(1 - probs + 1e-9))
        return np.concatenate([probs, conf, entropy], axis=1)

    X_val = make_features(model_results_val)
    X_test = make_features(model_results_test)

    meta_model = XGBClassifier(
        max_depth=3,
        learning_rate=0.1,
        n_estimators=100,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED,
        n_jobs=-1
    )

    meta_model.fit(X_val, y_val)
    y_pred_meta = meta_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred_meta)
    pr, rc, f1, _ = precision_recall_fscore_support(y_test, y_pred_meta, average="macro", zero_division=0)

    print(f" Meta-Ensemble Accuracy: {acc:.4f} | F1: {f1:.4f}")
    feature_importances = meta_model.feature_importances_
    print("Feature importances (first 10):", np.round(feature_importances[:10], 3))

    return {"name": "meta_ensemble_xgb", "acc": acc, "f1": f1}


# ---------- Main ----------
def main():
    device = "cuda" if (os.environ.get("USE_GPU","0")=="1" and torch.cuda.is_available()) else "cpu"
    print("[DEVICE]", device)

    emissions_data = load_emissions_data(EMISSIONS_FILE)

    print("[DATA] Loading IMDB 25% split…")
    df = load_imdb_dataframe()
    _, df_val, df_test = make_splits(df, SEED)

    val_results, test_results = [], []

    for m in MODELS:
        ckpt_dir = os.path.join(OUTDIR, m.replace("/", "_"), "checkpoint-1760")
        if not os.path.isdir(ckpt_dir):
            raise FileNotFoundError(f"Checkpoint folder not found: {ckpt_dir}")
        val_results.append(evaluate_model(ckpt_dir, df_val, emissions_data, device=device, split_name="val"))
        test_results.append(evaluate_model(ckpt_dir, df_test, emissions_data, device=device, split_name="test"))

    # XGBoost Meta-Ensemble
    meta_summary = stacked_ensemble_xgb(val_results, test_results)

    # Build combined summary table
    df_summary = pd.DataFrame([
        {"model": r["name"], "acc": r["acc"], "f1": r["f1"], "emissions": r["emissions"]} for r in test_results
    ])
    df_summary.loc[len(df_summary)] = {"model": meta_summary["name"], "acc": meta_summary["acc"], "f1": meta_summary["f1"], "emissions": sum(r["emissions"] for r in test_results)}

    # Plot
    plt.figure(figsize=(7, 4))
    plt.scatter(df_summary["emissions"], df_summary["acc"], s=100)
    for _, row in df_summary.iterrows():
        plt.text(row["emissions"] + 0.0001, row["acc"], row["model"], fontsize=9)
    plt.xlabel("CO₂ emissions (kg)")
    plt.ylabel("Accuracy")
    plt.title("RoBERTa, DistilBERT, BERT + Meta-Ensemble (XGBoost)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "meta_ensemble_xgb_plot.png"), dpi=150)
    plt.close()

    print("\n Results Summary:")
    print(df_summary.to_string(index=False, justify="center", float_format=lambda x: f"{x:.4f}"))

    df_summary.to_csv(os.path.join(OUTDIR, "results_summary_meta_xgb.csv"), index=False)
    print(" Saved summary to results_summary_meta_xgb.csv")

if __name__ == "__main__":
    main()