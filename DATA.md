# Data

Datasets are **not** committed to git (`data/` is ignored). This file explains how to
recreate them.

## Public datasets

### ASL Alphabet (Kaggle)

- **URL:** https://www.kaggle.com/datasets/grassknoted/asl-alphabet
- **Author:** Akash Nagaraj
- **Licence:** GPL-2.0 (as stated on the Kaggle page, checked 2026-09-30)
- **Downloaded:** 2026-09-30
- **Content:** ~87,000 200x200 images, 29 classes (A to Z, space, del, nothing).
  Largely one signer on similar backgrounds, so it is used **for training only**, never
  as the test set.

How to recreate:

1. Download the ZIP from the page above (Kaggle account needed).
2. Unzip so that `data/kaggle/asl_alphabet_train/asl_alphabet_train/A/*.jpg` exists.
   The `asl_alphabet_test` folder is not used.
3. Run `python scripts/download_models.py` (once), then
   `python scripts/extract_landmarks.py`.

This writes `data/processed/kaggle.npz` (landmarks only) and
`reports/kaggle_extraction.json` (how many images per letter had a detectable hand).
J and Z (motion letters) and the space/del/nothing classes are skipped.

Licence note: we never redistribute the images or extracted landmarks. If a model
trained on this data is ever published, publish it under a GPL-compatible licence.

## Own recordings

Recorded with:

```powershell
python scripts/record_landmarks.py --person p01 --session s01 --lighting "dnevno"
```

The page shows one letter at a time: SPACE records a 2-second clip, R redoes the
previous one, arrows skip. Default is 3 rounds through the 24 static letters (~5 min).

- Only **hand landmarks and metadata** are stored (person id, session id, lighting
  note), never images or video.
- Only people who agree are recorded; ids are pseudonymous (`p01`, `p02`, ...).
- Layout: `data/own/<person_id>/<session_id>/meta.json` and `<letter>_<n>.npz`
  (one file per clip: `points` (frames, 21, 3), `handedness` (frames,)).

## Evaluation split

Train/test is split **by person**, so the test set only contains people the model
has never seen.
