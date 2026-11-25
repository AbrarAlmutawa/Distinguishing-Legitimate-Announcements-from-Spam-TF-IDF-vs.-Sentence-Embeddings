"""Method 1: TF-IDF (word n-grams) + logistic regression.

All settings (ngram range, min_df, C, decision threshold) are chosen on the
DEV set only. The test set is never loaded in this file.

Usage:  python baseline_tfidf.py
Outputs: models/tfidf_lr.joblib, models/tfidf_lr_config.json, data/tfidf_dev_grid.csv
"""
import ast
import itertools
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

SEED = 42
GRID = {
    "ngram_range": [(1, 1), (1, 2)],
    "min_df": [1, 2, 3],
    "C": [0.1, 1, 10, 100],
}


def make_pipeline(ngram_range, min_df, C):
    """TF-IDF features (sublinear tf) followed by balanced logistic regression."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, sublinear_tf=True,
                                  ngram_range=ngram_range, min_df=min_df)),
        ("lr", LogisticRegression(C=C, max_iter=2000, class_weight="balanced",
                                  random_state=SEED)),
    ])


def grid_search_on_dev(train, dev):
    """Fit every grid combination on train, score on dev, return a ranked table."""
    rows = []
    for ngram, min_df, C in itertools.product(GRID["ngram_range"], GRID["min_df"], GRID["C"]):
        pipe = make_pipeline(ngram, min_df, C).fit(train["text"], train["label"])
        pred = pipe.predict(dev["text"])
        rows.append({
            "ngram_range": str(ngram), "min_df": min_df, "C": C,
            "dev_macro_f1": f1_score(dev["label"], pred, average="macro"),
            "dev_accuracy": accuracy_score(dev["label"], pred),
        })
    return pd.DataFrame(rows).sort_values(["dev_macro_f1", "dev_accuracy"], ascending=False)


def pick_threshold_on_dev(proba, y_dev):
    """Threshold that maximises dev macro F1 (ties -> higher threshold, fewer false positives)."""
    thresholds = np.round(np.arange(0.05, 0.96, 0.01), 2)
    scores = [f1_score(y_dev, (proba >= t).astype(int), average="macro") for t in thresholds]
    best = max(scores)
    return float(max(t for t, s in zip(thresholds, scores) if abs(s - best) < 1e-12)), best


def run(train_path="data/train.csv", dev_path="data/dev.csv", save=True, verbose=True):
    train, dev = pd.read_csv(train_path), pd.read_csv(dev_path)

    grid = grid_search_on_dev(train, dev)
    best = grid.iloc[0]
    ngram = ast.literal_eval(best["ngram_range"])
    pipe = make_pipeline(ngram, int(best["min_df"]), float(best["C"]))
    pipe.fit(train["text"], train["label"])

    proba = pipe.predict_proba(dev["text"])[:, 1]
    threshold, dev_f1 = pick_threshold_on_dev(proba, dev["label"])
    dev_pred = (proba >= threshold).astype(int)
    config = {
        "ngram_range": list(ngram), "min_df": int(best["min_df"]), "C": float(best["C"]),
        "class_weight": "balanced", "sublinear_tf": True, "threshold": threshold,
        "dev_macro_f1": round(dev_f1, 4),
        "dev_accuracy": round(float(accuracy_score(dev["label"], dev_pred)), 4),
        "dev_false_positives": int(((dev_pred == 1) & (dev["label"] == 0)).sum()),
        "dev_false_negatives": int(((dev_pred == 0) & (dev["label"] == 1)).sum()),
        "grid_combinations_tried": int(len(grid)), "seed": SEED,
    }
    if save:
        joblib.dump(pipe, "models/tfidf_lr.joblib")
        with open("models/tfidf_lr_config.json", "w") as f:
            json.dump(config, f, indent=2)
        grid.to_csv("data/tfidf_dev_grid.csv", index=False)
    if verbose:
        print(grid.head(8).to_string(index=False))
        print(config)
    return pipe, config, grid


if __name__ == "__main__":
    run()
