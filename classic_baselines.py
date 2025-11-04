# classic_baselines.py
# Simple one-by-one classical baselines for sentiment/semantic classification.
# Models: Logistic Regression, Linear SVM, Multinomial Naive Bayes.
# Usage:
#   python classic_baselines.py --data data.csv --text_col text --label_col label --model logreg
#   python classic_baselines.py --data data.csv --model svm
#   python classic_baselines.py --data data.csv --model nb
#
# CSV format: must contain at least two columns: text, label (0/1 or strings).
# Outputs: prints metrics, saves confusion matrix PNG and classification report CSV.

import argparse, os, sys, json
import numpy as np
import pandas as pd

from tensorflow.keras.datasets import imdb as keras_imdb

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, ConfusionMatrixDisplay, classification_report, confusion_matrix
import matplotlib.pyplot as plt

def _decode_imdb_review(encoded, reverse_index):
    # Keras IMDB reserves indices 0,1,2 for special tokens; shift by 3
    return " ".join([reverse_index.get(i - 3, "?") for i in encoded])

def load_imdb_dataframe(max_words: int = 10000, fraction: float = 0.1, seed: int = 42) -> pd.DataFrame:
    """
    Load IMDB dataset directly via Keras, decode to text, optionally sample a fraction, and
    return a DataFrame with columns ['text','label'].
    """
    num_words = None if (max_words is None or max_words <= 0) else max_words
    (X_train, y_train), (X_test, y_test) = keras_imdb.load_data(num_words=num_words)
    word_index = keras_imdb.get_word_index()
    reverse_word_index = {v: k for k, v in word_index.items()}

    all_X = list(X_train) + list(X_test)
    all_y = list(y_train) + list(y_test)
    texts = [_decode_imdb_review(seq, reverse_word_index) for seq in all_X]
    df = pd.DataFrame({"text": texts, "label": all_y})

    if 0 < fraction < 1.0:
        df = df.sample(frac=fraction, random_state=seed).reset_index(drop=True)
    return df

def build_pipeline(model_name: str, max_features: int = 50000, ngram_max: int = 2):
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents='unicode',
        max_features=max_features,
        ngram_range=(1, ngram_max),
    )
    if model_name == 'logreg':
        clf = LogisticRegression(max_iter=1000, n_jobs=None, class_weight='balanced')
    elif model_name == 'svm':
        clf = LinearSVC()  # fast and strong baseline
    elif model_name == 'nb':
        clf = MultinomialNB()
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return Pipeline([('tfidf', vectorizer), ('clf', clf)])

def encode_labels(y):
    # Map string labels to integers if needed
    if y.dtype == 'O':
        classes = sorted(y.unique())
        mapping = {c: i for i, c in enumerate(classes)}
        y_enc = y.map(mapping)
    else:
        mapping = None
        y_enc = y
    return y_enc, mapping

def main():
    ap = argparse.ArgumentParser(description='Classical text baselines (TF-IDF + classifier)')
    ap.add_argument('--data', required=False, help='Path to CSV with columns [text,label] (optional if --imdb is used)')
    ap.add_argument('--text_col', default='text', help='Text column name')
    ap.add_argument('--label_col', default='label', help='Label column name')
    ap.add_argument('--model', default='all', choices=['all', 'logreg', 'svm', 'nb'])
    ap.add_argument('--test_size', type=float, default=0.1)
    ap.add_argument('--val_size', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--max_features', type=int, default=50000)
    ap.add_argument('--ngram_max', type=int, default=2)
    ap.add_argument('--outdir', default='outputs_classic')
    ap.add_argument('--imdb', action='store_true', help='Load IMDB dataset directly via Keras instead of CSV')
    ap.add_argument('--imdb_fraction', type=float, default=0.1, help='Fraction of IMDB to use if --imdb (0<frac<=1)')
    ap.add_argument('--imdb_max_words', type=int, default=10000, help='Top-N most frequent words to keep; use 0 for unlimited vocabulary')
    ap.add_argument('--save_imdb_csv', default=None, help='Optional path to save the loaded IMDB DataFrame as CSV')

    # If the user just runs the file without any CLI args, behave smartly:
    # No args: use full IMDB and run all three models with unlimited vocab
    if len(sys.argv) == 1:
        ap.set_defaults(imdb=True, imdb_fraction=1.0, imdb_max_words=0, model='all')

    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Load data: either IMDB directly or user-provided CSV
    if args.imdb:
        if not (0 < args.imdb_fraction <= 1.0):
            print("ERROR: --imdb_fraction must be in (0,1].", file=sys.stderr)
            sys.exit(1)
        print(f"[DATA] Loading IMDB via Keras (max_words={'ALL' if args.imdb_max_words <= 0 else args.imdb_max_words}, fraction={args.imdb_fraction})...")
        df = load_imdb_dataframe(max_words=args.imdb_max_words, fraction=args.imdb_fraction, seed=args.seed)
        # enforce standard column names
        args.text_col = 'text'
        args.label_col = 'label'
        if args.save_imdb_csv:
            df.to_csv(args.save_imdb_csv, index=False)
            print(f"[DATA] Saved IMDB sample to {args.save_imdb_csv}")
    else:
        if not args.data:
            print("ERROR: provide --data CSV path or use --imdb to load Keras IMDB.", file=sys.stderr)
            sys.exit(1)
        df = pd.read_csv(args.data)
        if args.text_col not in df.columns or args.label_col not in df.columns:
            print(f"ERROR: CSV must contain '{args.text_col}' and '{args.label_col}' columns.", file=sys.stderr)
            sys.exit(1)

    X = df[args.text_col].astype(str).fillna('')
    y = df[args.label_col]
    y, label_map = encode_labels(y)

    # first split train+val vs test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )
    # then split train vs val
    val_fraction = args.val_size / (1.0 - args.test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_fraction, random_state=args.seed, stratify=y_trainval
    )

    models_to_run = ['logreg', 'svm', 'nb'] if args.model == 'all' else [args.model]
    for mname in models_to_run:
        pipe = build_pipeline(mname, args.max_features, args.ngram_max)
        pipe.fit(X_train, y_train)

        def eval_split(name, Xs, ys):
            yp = pipe.predict(Xs)
            acc = accuracy_score(ys, yp)
            pr, rc, f1, _ = precision_recall_fscore_support(ys, yp, average='macro', zero_division=0)
            print(f"[{name}] acc={acc:.4f} | F1={f1:.4f} | precision={pr:.4f} | recall={rc:.4f}")
            return yp, {'acc': acc, 'f1': f1, 'precision': pr, 'recall': rc}

        print(f"Training done. Model={mname} | n={len(X_train)} train, {len(X_val)} val, {len(X_test)} test")
        ypv, mval = eval_split('VAL', X_val, y_val)
        ypt, mtest = eval_split('TEST', X_test, y_test)

        # Classification report
        report = classification_report(y_test, ypt, output_dict=True, zero_division=0)
        rep_df = pd.DataFrame(report).transpose()
        rep_path = os.path.join(args.outdir, f'report_{mname}.csv')
        rep_df.to_csv(rep_path, index=True)

        # Confusion matrix
        cm = confusion_matrix(y_test, ypt)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(values_format='d')
        plt.title(f'Confusion Matrix - {mname.upper()}')
        fig_path = os.path.join(args.outdir, f'cm_{mname}.png')
        plt.savefig(fig_path, bbox_inches='tight', dpi=150)
        plt.close()

        # Save brief summary JSON
        summary = {
            'model': mname,
            'val': mval,
            'test': mtest,
            'label_map': label_map,
            'n_train': int(len(X_train)), 'n_val': int(len(X_val)), 'n_test': int(len(X_test)),
            'params': {'max_features': args.max_features, 'ngram_max': args.ngram_max}
        }
        with open(os.path.join(args.outdir, f'summary_{mname}.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Saved: {rep_path}, {fig_path}")

    # After loop, return to end of function
    return

if __name__ == '__main__':
    main()
