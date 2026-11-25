# Legitimate Academic Announcements vs. Spam (Ling-Spam)

NLP course project, Option 1 (Spam Detection). Compares **TF-IDF + logistic regression** with **pretrained sentence embeddings (all-MiniLM-L6-v2) + logistic regression**, with a focus on false positives for legitimate academic announcements that contain promotional vocabulary.

**Runtime:** Google Colab (also works in local Jupyter). Label convention: `0` = Not spam, `1` = Spam.

## Quick start (Colab)
1. Open `ling_spam_pipeline.ipynb` in Colab (File > Open notebook > GitHub, or upload it).
2. In the Setup cell, set `REPO_URL` to this repository's URL. The cell installs the pinned packages and clones the repo so all relative paths work.
3. Runtime > Restart session and run all, then check that every cell finishes without errors.
4. The **Try it** section loads the saved models from `models/`; nothing is retrained.

## Quick start (local)
```bash
git clone https://github.com/AbrarAlmutawa/Distinguishing-Legitimate-Announcements-from-Spam-TF-IDF-vs.-Sentence-Embeddings && cd Distinguishing-Legitimate-Announcements-from-Spam-TF-IDF-vs.-Sentence-Embeddings
pip install -r requirements.txt
jupyter notebook ling_spam_pipeline.ipynb
```

## Repository layout
| Path | Purpose |
|---|---|
| `ling_spam_pipeline.ipynb` | Main notebook: Setup, Data, Method 1, Method 2, Results, Try it, Sources |
| `data/messages.csv` | Raw Ling-Spam (Kaggle version): `subject`, `message`, `label` |
| `data/train.csv`, `dev.csv`, `test.csv` | Leak-free 70/15/15 splits (created by the notebook or `prepare_data.py`) |
| `data/dedup_split_report.json` | Counts removed/grouped and the leakage check |
| `data_utils.py` | Loading, normalising, de-duplication, group-aware split, leakage check |
| `prepare_data.py` | Runs the data pipeline from the command line |
| `baseline_tfidf.py` | Method 1: tuning on dev and saving the model |
| `models/` | Saved models (`tfidf_lr.joblib`, config JSON; Method 2 files added by Jana) |
| `report/` | Draft report sections, appendix entry, slide text |

## Method summary
- **Input text:** `subject + " " + message`.
- **Cleaning:** lowercase and whitespace-collapsed copy used for duplicate checks; exact duplicates removed; near-duplicates (TF-IDF cosine > 0.9) grouped and kept inside one split.
- **Tuning:** all settings chosen on the dev set; the test set is evaluated once at the end.
- **Seed:** 42 everywhere.

## Credits
Ling-Spam corpus (Androutsopoulos et al., 2000), Kaggle copy; scikit-learn; Sentence-BERT (Reimers and Gurevych, 2019); AI assistance from Claude (Anthropic) for drafting code, reviewed and run by the team.
