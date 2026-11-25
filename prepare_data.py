"""Run the full data pipeline: load -> verify -> dedup -> group split -> save.

Usage:  python prepare_data.py
Outputs: data/train.csv, data/dev.csv, data/test.csv, data/dedup_split_report.json
"""
import data_utils as du

raw = du.load_raw("data/messages.csv")
report = {
    "raw_rows": int(len(raw)),
    "raw_spam": int((raw["label"] == 1).sum()),
    "raw_legit": int((raw["label"] == 0).sum()),
    "rows_with_empty_subject": int((raw["subject"].str.strip() == "").sum()),
    "input_text": "subject + ' ' + message (empty subject -> body only)",
}

clean, exact_info = du.drop_exact_duplicates(raw)
report.update(exact_info)
report["rows_after_exact_dedup"] = int(len(clean))

clean["group"], near_info = du.near_duplicate_groups(clean)
report.update(near_info)

clean["split"] = du.grouped_stratified_split(clean)
parts = {s: clean[clean["split"] == s].reset_index(drop=True) for s in ("train", "dev", "test")}
du.check_no_leakage(parts["train"], parts["dev"], parts["test"])
report["leakage_check"] = "passed: no group id or normalised text shared across train/dev/test"
report["splits"] = du.summarise_splits(parts["train"], parts["dev"], parts["test"])
report["seed"] = du.SEED

cols = ["subject", "message", "text", "label", "group"]
for name, part in parts.items():
    part[cols].to_csv(f"data/{name}.csv", index=False)

du.save_report(report)
for k, v in report.items():
    print(f"{k}: {v}")
