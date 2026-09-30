# instruction.md: Sign language fingerspelling trainer (Smotra project)

Read this whole file before doing anything. It is the agreed plan for this repository. If something here conflicts with what the user asks later, follow the user, then update this file so it stays true.

## 0. Working agreements

- Talk to the user in **Croatian**. Write **code, comments, docstrings, commit messages, and the README in English**.
- The user is a 4th-year FER student. She is competent but wants to understand what gets built, so explain non-obvious choices briefly and don't dump giant unexplained files.
- Work **one milestone at a time** (section 6). Don't scaffold the whole project at once. After each milestone, stop, summarize what works, and ask before moving on.
- Ask before making large decisions (UI framework, model architecture change, new dependencies). Small decisions: just decide and mention them.
- Everything must run on a **normal Windows laptop with a webcam, without a GPU and without internet** (the demo runs at a fair stand with unreliable Wi-Fi). Only Windows needs to be supported.
- The user works alone (no mentor, no co-author).
- Keep it simple. A working, polished demo beats an ambitious half-working one.

## 1. Context and goal

The user is applying to **Smotra Sveučilišta u Zagrebu** (University of Zagreb fair, held in November) with a project representing FER. A FER committee picks the works. Their criteria are: **attractiveness, innovation, good promotion of the profession, and ability to present at the fair**. The prize is 700 EUR. Past accepted works are mostly demos a visitor can try in about 30 seconds (barcode price scanner, VR chess, hand-controlled flight, waste-sorting app, etc.).

Audience at the fair: mostly **high-school students and the general public**. They should be able to walk up, try it, and understand it without explanation.

**Project:** a real-time, webcam-based **fingerspelling trainer** for **American Sign Language (ASL)**, built in Python, extended with a **small fixed set of common phrase signs** (e.g. HELLO, THANK YOU, I LOVE YOU, GOODBYE) and **live subtitles**. Hrvatski znakovni jezik (HZJ) fingerspelling is a stretch goal, not the main deliverable.

Target experience: the visitor stands in front of the webcam and signs. Each recognised letter or sign appears immediately in a subtitle bar at the bottom of the screen. When the visitor pauses, the recognised words are matched against a local table of predefined phrases and the full sentence is shown (e.g. signs THANK YOU -> subtitle "Thank you."). Later, a reference image or GIF of the last recognised sign is shown in the bottom-right corner, above the subtitles.

Important positioning: "webcam recognizes ASL letters" is a very common tutorial project. To stand out, this project must be presented as an **interactive learning and spelling experience with its own trained and evaluated model**, not as a plain recogniser.

## 2. Scope decisions (already agreed)

**In scope**
- Python only. No robots, no VR.
- ASL fingerspelling (letters A to Z) from a live webcam. Letters come first.
- After letters: a small, closed vocabulary of common ASL phrase signs (roughly 10 to 20), recognised from short landmark sequences.
- Live subtitles: words appear as they are recognised; after a pause the phrase is completed into a sentence from a **local, predefined phrase table** (no internet, no language model).
- A printable list of everything the program can recognise (`docs/SIGNS.md`, generated from the vocabulary config) for the written paper.
- Reference PNG/GIF of the current sign in the bottom-right corner, above the subtitles. **Only after subtitles work** and once the user has collected the assets.
- Own model trained on hand landmarks, with honest evaluation.
- Interactive modes for visitors (section 4).
- Language as a configuration parameter, so HZJ can be added later without code changes.

**Out of scope**
- Open-vocabulary sign recognition or real translation of arbitrary signed sentences. Only the fixed vocabulary and the predefined phrases are supported; present it that way.
- Claiming to "translate HZJ". The HZJ stage, if done, is only "HZJ fingerspelling alphabet".
- Mobile app, cloud backend, accounts.

## 3. Technical approach

**Pipeline:** webcam frame -> MediaPipe hand landmarks (21 points) -> normalisation -> classifier -> smoothing/confirmation logic -> UI.

**Suggested stack** (confirm with the user before adding anything else):
- Python 3.11+
- `mediapipe` (HandLandmarker), `opencv-python`, `numpy`
- `scikit-learn` for the baseline; a small `torch` MLP only if it clearly helps
- Simple UI: start with an OpenCV window with overlays. Discuss a nicer UI (e.g. pygame or a local web UI) only after the core works.
- `pytest` for the feature-normalisation and data-handling code

**Landmark normalisation** (this is where most accuracy comes from):
- translate so the wrist is the origin
- scale by a hand-size measure (e.g. wrist to middle-finger MCP distance)
- mirror left hands to right (or train on both), document the choice
- flatten to a feature vector (and optionally add pairwise distances / finger angles)

**Models:** start with a simple baseline (logistic regression / random forest / small MLP). Compare at least two. Report accuracy, per-class accuracy, and a confusion matrix.

**Evaluation must be honest:** split **by person/session**, not randomly by frame, so the test set contains people the model has not seen. State this in the README. Test under different lighting and backgrounds, and report what breaks.

**Static vs dynamic letters:** most ASL letters are static poses. **J and Z involve motion.** Handle them after the static letters work, using a short window of landmark frames (for example a small 1D-CNN/GRU, or a trajectory heuristic on the fingertip). Until then, exclude J and Z and say so clearly in the UI and docs.

**Smoothing and confirmation:** never output a letter from a single frame. Use a rolling window / majority vote plus a "hold for about 0.7 s to confirm" rule. Show a confidence indicator. Show "no hand detected" when appropriate.

## 4. Product features (visitor-facing modes)

1. **Free spelling:** visitor signs letters, the app builds a word on screen, with visual feedback on confidence and confirmation progress.
2. **Sign your name:** visitor types or picks a name, the app shows the reference sign for each letter in order and gives live feedback ("good", "adjust your thumb", or at least "not recognised yet"). This is the main crowd-pleaser, so make it smooth.
3. **Word suggestions:** simple autocomplete from a word list while spelling (frequency-based; no heavy NLP needed).
4. **Live subtitles and phrases:** recognised letters/signs appear in a subtitle bar as they come; after a pause they are matched to a predefined phrase and shown as a full sentence.
5. **Language switch (config):** `asl` now, `hzj` later.

Reference images/GIFs (for "sign your name" and the corner preview) may be our own or publicly available ones that are free to use. The user finds them; Claude helps check whether each one's licence allows use. Do not use assets whose licence is unclear or forbids it. Record the source URL and licence of every asset in `ASSETS.md`.

**Sign references:** the user has no ASL speaker to check with. Her sources are YouTube and ASL portals (e.g. Lifeprint, Handspeak). That is accepted; don't keep raising it. Record the reference source for each sign in the vocabulary config so it appears in `docs/SIGNS.md`.

## 5. Data

- **Public ASL data:** the ASL Alphabet dataset on Kaggle (image-based) can be run through MediaPipe to extract landmarks. **Check the licence and terms before use** and record them in `DATA.md`.
- **Own data collection:** write a small recorder script (`scripts/record_landmarks.py`) that captures landmarks from the webcam per letter, per person, per session, and stores **only landmarks plus metadata** (person id, session id, lighting note), **not face video or images**.
- Target: a few hundred samples per letter, from several different people. Own data matters for robustness to the actual fair setting.
- **Privacy and consent:** only record people who agree. Store pseudonymous ids. Never commit raw data to git.
- Keep datasets out of git (`data/` in `.gitignore`). Keep a `DATA.md` explaining how to recreate them.

## 6. Milestones

Stop after each one and report.

**M0: Repo setup**
- Git repo, `README.md`, `.gitignore`, `pyproject.toml` or `requirements.txt`, package layout, `pytest` running, `DATA.md`, `ASSETS.md`.
- Short README: what it is, how to install, how to run.

**M1: Live landmarks**
- Webcam capture with MediaPipe landmarks drawn on screen, running at a smooth frame rate.
- Landmark normalisation function with unit tests.

**M2: Data pipeline and baseline model**
- Landmark extraction from the public dataset.
- Recorder script for own data.
- Baseline classifier for static letters (A to Z without J and Z).
- Evaluation script: per-person split, accuracy, confusion matrix saved to `reports/`.

**M3: Real-time recognition**
- Live prediction with smoothing, hold-to-confirm, and confidence display.
- Free spelling mode working end to end, with letters appearing in a bottom subtitle bar.

**M4: Visitor experience**
- "Sign your name" mode with reference images and feedback.
- Word suggestions.
- Clean, readable UI that works from 1 to 2 metres away on a laptop screen. Large fonts, high contrast, obvious instructions (assume the visitor reads nothing).

**M5: Robustness and fair readiness**
- Test with several people, different lighting, left and right hands.
- Offline run verified. One-command startup (`python -m signtrainer` or a script).
- Fallback behaviour if the camera fails.
- Short demo script (30-second visitor flow) in `docs/DEMO.md`.

**M6: Dynamic signs (J, Z and phrase signs)**
- Sequence-based recognition (short window of landmark frames; hands, plus face/pose reference points where a sign needs them, e.g. THANK YOU starts at the chin).
- Closed vocabulary of common phrase signs: HELLO, THANK YOU, I LOVE YOU, GOODBYE, and similar, agreed with the user.
- Honest reporting of how well it works.

**M7: Subtitles and phrases**
- Words appear in the subtitle bar as soon as they are recognised; after a pause the sequence is matched against a local phrase table (e.g. `phrases.yaml`) and shown as a full sentence.
- Generate `docs/SIGNS.md`: every recognisable letter, sign and phrase, with its reference source, ready to print for the paper.

**M8: Reference sign preview**
- PNG or GIF of the last recognised sign in the bottom-right corner, above the subtitles.
- Only after M7 works and the user has collected assets with checked licences (`ASSETS.md`).

**M9: Stretch, HZJ fingerspelling alphabet**
- Only after M1 to M5 are solid.
- Language selected via config. Code must not need changes for a new alphabet, only new data, classes, and reference assets.
- **Sign accuracy must be verified by a qualified source.** Do not invent or guess Croatian signs from memory. Ask the user to check with the Hrvatski savez gluhih i nagluhih or the Edukacijsko-rehabilitacijski fakultet, and to have a community member review the demo before the fair.
- Present it as "HZJ fingerspelling alphabet", not as a translator.

**M10: Application and presentation material**
- Help draft the Smotra application text (Croatian), a one-page project summary, a poster outline, and screenshots or a short demo video. The application must emphasise the learning experience, the own trained and evaluated model, the accessibility angle, and how it promotes the profession.

## 7. Suggested repository layout

```
.
├── README.md
├── instruction.md
├── DATA.md
├── ASSETS.md
├── pyproject.toml
├── src/signtrainer/
│   ├── __init__.py
│   ├── capture.py        # webcam + MediaPipe
│   ├── features.py       # normalisation, feature vectors
│   ├── model.py          # train / load / predict
│   ├── smoothing.py      # rolling vote, hold-to-confirm
│   ├── modes/            # free_spelling.py, sign_name.py
│   ├── ui.py
│   └── config.py         # language, thresholds, paths
├── scripts/
│   ├── record_landmarks.py
│   ├── extract_landmarks.py
│   └── evaluate.py
├── tests/
├── models/               # small trained models (check size; consider releases if large)
├── reports/              # metrics, confusion matrices
├── assets/               # licensed or self-made reference images
├── data/                 # gitignored
└── docs/DEMO.md
```

## 8. Git workflow

- Small, frequent commits using **Conventional Commits** (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`).
- One branch per milestone (`m1-live-landmarks`, ...), merged to `main` when it works. Open a GitHub issue per milestone if the user wants.
- Never commit: raw datasets, recorded personal data, API keys, large binaries. Check `git status` before each commit.
- Tag a working version before the fair (for example `v1.0-smotra`).
- Keep the README current: install steps, how to run the demo, how to retrain, limitations.

## 9. Quality bar and definition of done

The project is ready when:
- A stranger can start it with one command and use it without instructions.
- Letter recognition is stable enough that the "sign your name" flow works for most people in fair conditions.
- Evaluation numbers in the README are real, use a person-independent split, and include limitations.
- It runs offline on a laptop.
- Licences and sources for all data and assets are documented.
- The demo works with a different person than the one who built it.

## 10. Open questions to raise with the user

- Smotra application deadline and the exact submission format (check the FER intranet announcement).
- What laptop and camera will be used at the stand (Windows is confirmed).
- UI choice: OpenCV window versus something prettier, decided after M3.
- Exact phrase vocabulary for M6/M7 (start: HELLO, THANK YOU, I LOVE YOU, GOODBYE).

Resolved: no mentor (the user works alone); Windows only; sign references from YouTube/ASL portals are accepted.
