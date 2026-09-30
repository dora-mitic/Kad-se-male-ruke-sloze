# Data

Datasets are **not** committed to git (`data/` is ignored). This file explains how to
recreate them.

## Public datasets

None downloaded yet. For every public dataset, record here: name, URL, licence,
date downloaded, and how landmarks were extracted.

## Own recordings

Recorded with `scripts/record_landmarks.py` (added in M2).

- Only **hand landmarks and metadata** are stored (person id, session id, lighting
  note), never images or video.
- Only people who agree are recorded; ids are pseudonymous (`p01`, `p02`, ...).
- Layout: `data/own/<person_id>/<session_id>/<label>.npy` (may change in M2).

## Evaluation split

Train/test is split **by person**, so the test set only contains people the model
has never seen.
