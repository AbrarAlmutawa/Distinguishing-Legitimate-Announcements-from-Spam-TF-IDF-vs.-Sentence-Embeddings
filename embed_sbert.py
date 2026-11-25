"""Method 2: pretrained sentence embeddings (all-MiniLM-L6-v2) + logistic regression.

Owner: Jana.

Pipeline:  text -> SentenceTransformer("all-MiniLM-L6-v2") -> 384-d L2-normalised
embedding -> logistic regression -> P(spam) -> threshold -> label.

All settings (C and the decision threshold) are chosen on the DEV set only.
The test set is never loaded in this file; the Results section calls
`predict_proba_texts` on it once, after both models are final.

Notes
- The encoder is used as-is (frozen pretrained weights); only the logistic
  regression is trained.
- all-MiniLM-L6-v2 reads at most 256 word-piece tokens, so each email is
  represented by its subject and the start of its body. Text after that is
  ignored (a documented limitation; see the report).
- Embeddings are cached in data/emb_cache/ keyed by a hash of the texts, so
  re-running the notebook does not re-encode unless the data changes.

Usage:  python embed_sbert.py
Outputs: models/sbert_lr.joblib, models/sbert_lr_config.json, data/sbert_dev_grid.csv
"""
import hashlib
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

SEED = 42
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 64
CACHE_DIR = "data/emb_cache"
C_GRID = [0.01, 0.1, 1, 10, 100]

_encoder = None  # loaded once, reused by every call


def get_encoder():
    """Load the pretrained sentence encoder once (downloads ~90 MB on first use)."""
    global _encoder
    if _encoder is None:
        import torch
        from sentence_transformers import SentenceTransformer
        torch.manual_seed(SEED)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _encoder = SentenceTransformer(MODEL_NAME, device=device)
    return _encoder


def encoder_revision():
    """Hugging Face commit hash of the loaded weights (recorded for reproducibility)."""
    try:
        return get_encoder()[0].auto_model.config._commit_hash
    except Exception:
        return None


def embed_texts(texts, cache_name=None):
    """Encode a list of texts into L2-normalised embeddings, shape (n, 384).

    If cache_name is given, the result is cached on disk and reused when the
    exact same texts are encoded again.
    """
    texts = [str(t) for t in texts]
    path = None
    if cache_name:
        digest = hashlib.sha1("\x00".join(texts).encode("utf-8")).hexdigest()[:12]
        path = os.path.join(CACHE_DIR, f"{cache_name}_{digest}.npy")
        if os.path.exists(path):
            return np.load(path)
    emb = get_encoder().encode(texts, batch_size=BATCH_SIZE, convert_to_numpy=True,
                               normalize_embeddings=True, show_progress_bar=bool(cache_name))
    emb = emb.astype(np.float32)
    if path:
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.save(path, emb)
    return emb


def make_classifier(C):
    """Logistic regression with balanced class weights (spam is ~16% of the data)."""
    return LogisticRegression(C=C, max_iter=5000, class_weight="balanced", random_state=SEED)


def grid_search_on_dev(X_train, y_train, X_dev, y_dev):
    """Fit one classifier per C on train, score on dev, return a ranked table."""
    rows = []
    for C in C_GRID:
        pred = make_classifier(C).fit(X_train, y_train).predict(X_dev)
        rows.append({
            "C": C,
            "dev_macro_f1": f1_score(y_dev, pred, average="macro"),
            "dev_accuracy": accuracy_score(y_dev, pred),
        })
    # ties -> smaller C (stronger regularisation, simpler model)
    return pd.DataFrame(rows).sort_values(["dev_macro_f1", "dev_accuracy", "C"],
                                          ascending=[False, False, True])


def pick_threshold_on_dev(proba, y_dev):
    """Threshold that maximises dev macro F1 (ties -> higher threshold, fewer false positives).

    Same rule as Method 1, so the two models are tuned in the same way.
    """
    thresholds = np.round(np.arange(0.05, 0.96, 0.01), 2)
    scores = [f1_score(y_dev, (proba >= t).astype(int), average="macro") for t in thresholds]
    best = max(scores)
    return float(max(t for t, s in zip(thresholds, scores) if abs(s - best) < 1e-12)), best


def run(train_path="data/train.csv", dev_path="data/dev.csv", save=True, verbose=True):
    train, dev = pd.read_csv(train_path), pd.read_csv(dev_path)
    X_train = embed_texts(train["text"], cache_name="train")
    X_dev = embed_texts(dev["text"], cache_name="dev")
    y_train, y_dev = train["label"].to_numpy(), dev["label"].to_numpy()

    grid = grid_search_on_dev(X_train, y_train, X_dev, y_dev)
    best_C = float(grid.iloc[0]["C"])
    clf = make_classifier(best_C).fit(X_train, y_train)

    proba = clf.predict_proba(X_dev)[:, 1]
    threshold, dev_f1 = pick_threshold_on_dev(proba, y_dev)
    dev_pred = (proba >= threshold).astype(int)
    config = {
        "encoder": MODEL_NAME,
        "encoder_revision": encoder_revision(),
        "embedding_dim": int(X_train.shape[1]),
        "max_seq_length": int(getattr(get_encoder(), "max_seq_length", 256)),
        "normalize_embeddings": True,
        "C": best_C, "class_weight": "balanced", "threshold": threshold,
        "dev_macro_f1": round(dev_f1, 4),
        "dev_accuracy": round(float(accuracy_score(y_dev, dev_pred)), 4),
        "dev_false_positives": int(((dev_pred == 1) & (y_dev == 0)).sum()),
        "dev_false_negatives": int(((dev_pred == 0) & (y_dev == 1)).sum()),
        "grid_combinations_tried": int(len(grid)), "seed": SEED,
    }
    if save:
        os.makedirs("models", exist_ok=True)
        joblib.dump(clf, "models/sbert_lr.joblib")
        with open("models/sbert_lr_config.json", "w") as f:
            json.dump(config, f, indent=2)
        grid.to_csv("data/sbert_dev_grid.csv", index=False)
    if verbose:
        print(grid.to_string(index=False))
        print(config)
    return clf, config, grid


# --- Loading the saved model (used by Results and Try it; no retraining) ------
def load_saved(model_path="models/sbert_lr.joblib", config_path="models/sbert_lr_config.json"):
    with open(config_path) as f:
        cfg = json.load(f)
    return joblib.load(model_path), cfg


def predict_proba_texts(texts, clf=None, cfg=None):
    """P(spam) for a list of texts using the saved classifier."""
    if clf is None:
        clf, cfg = load_saved()
    return clf.predict_proba(embed_texts(texts))[:, 1]


def predict_texts(texts, clf=None, cfg=None):
    """0/1 predictions for a list of texts using the saved classifier and dev threshold."""
    if clf is None:
        clf, cfg = load_saved()
    return (predict_proba_texts(texts, clf, cfg) >= cfg["threshold"]).astype(int)


if __name__ == "__main__":
    run()
