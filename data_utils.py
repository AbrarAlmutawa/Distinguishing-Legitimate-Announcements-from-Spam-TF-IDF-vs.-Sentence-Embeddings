"""Data loading, cleaning, de-duplication and leak-free splitting for Ling-Spam.

Used by the notebook (Setup -> Data) and by prepare_data.py.
Label convention in messages.csv: 0 = legitimate (Not spam), 1 = spam.
"""
import re
import json
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedGroupKFold

SEED = 42
NEAR_DUP_THRESHOLD = 0.9
LABEL_NAMES = {0: "Not spam", 1: "Spam"}


def normalise(text: str) -> str:
    """Lowercase and collapse all whitespace runs into one space."""
    return re.sub(r"\s+", " ", str(text).lower()).strip()


def build_input_text(subject, message) -> str:
    """Model input = subject + " " + body (missing subject becomes empty)."""
    subject = "" if pd.isna(subject) else str(subject)
    message = "" if pd.isna(message) else str(message)
    return f"{subject} {message}".strip()


def load_raw(path="data/messages.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    expected = {"subject", "message", "label"}
    assert expected.issubset(df.columns), f"missing columns: {expected - set(df.columns)}"
    df["subject"] = df["subject"].fillna("")
    df["text"] = [build_input_text(s, m) for s, m in zip(df["subject"], df["message"])]
    df["text_norm"] = df["text"].map(normalise)
    return df


def drop_exact_duplicates(df: pd.DataFrame):
    """Remove exact duplicates of the normalised text (keep first occurrence)."""
    dup = df.duplicated("text_norm", keep="first")
    conflicts = int((df.groupby("text_norm")["label"].nunique() > 1).sum())
    removed = df[dup]
    out = df[~dup].reset_index(drop=True)
    info = {
        "exact_duplicates_removed": int(dup.sum()),
        "removed_spam": int((removed["label"] == 1).sum()),
        "removed_legit": int((removed["label"] == 0).sum()),
        "exact_duplicate_texts_with_conflicting_labels": conflicts,
    }
    return out, info


def near_duplicate_groups(df: pd.DataFrame, threshold=NEAR_DUP_THRESHOLD):
    """Group messages whose TF-IDF cosine similarity exceeds the threshold.

    Connected components over the similarity graph: if A~B and B~C all three
    share one group id, so the whole group can be kept inside one split.
    """
    X = TfidfVectorizer(sublinear_tf=True).fit_transform(df["text_norm"])
    sim = (X @ X.T).tocsr()
    adj = sim.copy()
    adj.data = (adj.data > threshold).astype(np.int8)
    adj.setdiag(0)
    adj.eliminate_zeros()
    _, group = connected_components(adj, directed=False)
    sizes = np.bincount(group)
    info = {
        "near_dup_threshold": threshold,
        "near_dup_groups_with_2plus_messages": int((sizes > 1).sum()),
        "messages_in_near_dup_groups": int(sizes[sizes > 1].sum()),
        "largest_group_size": int(sizes.max()),
    }
    return group, info


def grouped_stratified_split(df: pd.DataFrame, group_col="group", seed=SEED):
    """Stratified 70/15/15 split in which every group stays in one partition.

    StratifiedGroupKFold with 20 folds: 14 folds -> train, 3 -> dev, 3 -> test.
    """
    sgkf = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=seed)
    fold = np.empty(len(df), dtype=int)
    for k, (_, idx) in enumerate(sgkf.split(df, df["label"], df[group_col])):
        fold[idx] = k
    split = np.where(fold < 14, "train", np.where(fold < 17, "dev", "test"))
    return pd.Series(split, index=df.index, name="split")


def check_no_leakage(train, dev, test):
    """Raise AssertionError if any group or normalised text crosses partitions."""
    parts = {"train": train, "dev": dev, "test": test}
    names = list(parts)
    for i in range(3):
        for j in range(i + 1, 3):
            a, b = parts[names[i]], parts[names[j]]
            assert not set(a["group"]) & set(b["group"]), f"group overlap {names[i]}/{names[j]}"
            assert not set(a["text_norm"]) & set(b["text_norm"]), f"text overlap {names[i]}/{names[j]}"
    return True


def summarise_splits(train, dev, test) -> dict:
    out = {}
    for name, part in (("train", train), ("dev", dev), ("test", test)):
        out[name] = {
            "n": int(len(part)),
            "spam": int((part["label"] == 1).sum()),
            "legit": int((part["label"] == 0).sum()),
            "spam_share": round(float((part["label"] == 1).mean()), 4),
        }
    return out


def save_report(report: dict, path="data/dedup_split_report.json"):
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
