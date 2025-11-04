import argparse, os, json, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.datasets import imdb as keras_imdb

def _decode_imdb_review(encoded, reverse_index):
    # Keras IMDB reserves indices 0,1,2 for special tokens; shift by 3
    return " ".join([reverse_index.get(i - 3, "?") for i in encoded])

def load_imdb_dataframe(max_words: int = 10000, fraction: float = 1.0, seed: int = 42) -> pd.DataFrame:
    """
    Load IMDB dataset directly via Keras, decode to text, optionally sample a fraction, and
    return a DataFrame with columns ['text','label'].
    If max_words <= 0, uses unlimited vocabulary.
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

def main():
    ap = argparse.ArgumentParser(description='Embedding + LSTM baseline')
    ap.add_argument('--data', required=False, help='CSV path with [text,label] (optional if --imdb is used)')
    ap.add_argument('--text_col', default='text')
    ap.add_argument('--label_col', default='label')
    ap.add_argument('--max_words', type=int, default=20000, help='Tokenizer vocabulary size; use 0 for unlimited')
    ap.add_argument('--max_len', type=int, default=128)
    ap.add_argument('--embed_dim', type=int, default=128)
    ap.add_argument('--lstm_units', type=int, default=128)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--epochs', type=int, default=6)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--outdir', default='outputs_lstm')
    # IMDB direct loading flags
    ap.add_argument('--imdb', action='store_true', help='Load IMDB dataset directly via Keras instead of CSV')
    ap.add_argument('--imdb_fraction', type=float, default=1.0, help='Fraction of IMDB to use if --imdb (0<frac<=1)')
    ap.add_argument('--imdb_max_words', type=int, default=0, help='Top-N most frequent words to keep; 0 = unlimited vocabulary')
    ap.add_argument('--save_imdb_csv', default=None, help='Optional path to save the loaded IMDB DataFrame as CSV')

    # If the user just runs the file without any CLI args, behave smartly:
    # default to IMDB FULL and run.
    if len(sys.argv) == 1:
        ap.set_defaults(imdb=True, imdb_fraction=1.0, imdb_max_words=0)

    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Load data: either IMDB directly or user-provided CSV
    if args.imdb:
        if not (0 < args.imdb_fraction <= 1.0):
            print("ERROR: --imdb_fraction must be in (0,1].", file=sys.stderr)
            sys.exit(1)
        print(f"[DATA] Loading IMDB via Keras (max_words={'ALL' if args.imdb_max_words <= 0 else args.imdb_max_words}, fraction={args.imdb_fraction})...")
        df = load_imdb_dataframe(max_words=args.imdb_max_words, fraction=args.imdb_fraction, seed=args.seed)
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
    y_raw = df[args.label_col].astype(str)

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.1, random_state=args.seed, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.1111, random_state=args.seed, stratify=y_trainval)  # ~10% val

    tok = Tokenizer(num_words=(None if args.max_words <= 0 else args.max_words), oov_token="<OOV>")
    tok.fit_on_texts(X_train)
    def to_seq(series):
        return pad_sequences(tok.texts_to_sequences(series), maxlen=args.max_len, padding='post', truncating='post')
    Xtr = to_seq(X_train); Xva = to_seq(X_val); Xte = to_seq(X_test)

    vocab_size = (args.max_words if args.max_words > 0 else (len(tok.word_index) + 1))
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=args.embed_dim, input_length=args.max_len),
        LSTM(args.lstm_units, dropout=0.2, recurrent_dropout=0.2),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    es = EarlyStopping(monitor='val_accuracy', patience=2, restore_best_weights=True)
    hist = model.fit(Xtr, y_train, validation_data=(Xva, y_val),
                     epochs=args.epochs, batch_size=args.batch_size, callbacks=[es], verbose=2)

    # Evaluate
    loss, acc = model.evaluate(Xte, y_test, verbose=0)
    metrics = {'test_acc': float(acc), 'test_loss': float(loss)}
    with open(os.path.join(args.outdir, 'metrics_lstm.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    # Curves
    plt.figure()
    plt.plot(hist.history['accuracy'], label='train_acc')
    plt.plot(hist.history['val_accuracy'], label='val_acc')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.title('LSTM Accuracy')
    plt.savefig(os.path.join(args.outdir, 'lstm_acc.png'), dpi=150, bbox_inches='tight'); plt.close()

    plt.figure()
    plt.plot(hist.history['loss'], label='train_loss')
    plt.plot(hist.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.title('LSTM Loss')
    plt.savefig(os.path.join(args.outdir, 'lstm_loss.png'), dpi=150, bbox_inches='tight'); plt.close()

    # Save label mapping
    with open(os.path.join(args.outdir, 'label_mapping.json'), 'w') as f:
        json.dump({'classes_': le.classes_.tolist()}, f, indent=2)

    print(f"Done. Test acc={acc:.4f}. Outputs saved in {args.outdir}")
if __name__ == '__main__':
    main()
