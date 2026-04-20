<div align="center">

```
____  _           _          _
| __ )(_) __ _ ___| |    __ _| |__
|  _ \| |/ _` / __| |   / _` | '_ \
| |_) | | (_| \__ \ |__| (_| | |_) |
|____/|_|\__,_|___/_____\__,_|_.__/

         Media  Bias  Radar
```

### Context-Aware News Analysis System.

Paste an article, get a six-axis **Bias Radar**, sentence-level evidence summaries, PDF reporting, and uncertainty tracking. BiasLab transcends simple word counting by analyzing sentences in context — properly ignoring quoted speech and accounting for negations to eliminate false positives. Runs 100% locally.

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Tkinter](https://img.shields.io/badge/GUI-Tkinter-FFC107?style=for-the-badge)
![NLTK](https://img.shields.io/badge/NLP-NLTK-2ca02c?style=for-the-badge)
![License](https://img.shields.io/badge/license-MIT-green?style=for-the-badge)

</div>

---

## What is this?

**BiasLab** is a Python desktop app that analyzes a news article and measures how neutral its language is. You paste text, click **Analyze**, and you get:

* An **overall bias score** from 0 (reads like straight reporting) to 100 (reads like opinion/propaganda).
* A **six-axis Bias Radar** chart visualizing *where* the bias comes from.
* A **LEFT <-> RIGHT political lean** number (-1.00 to +1.00) separate from the intensity score.
* A **Confidence score & Uncertainty Zone** that explicitly tells you how reliable the score is based on signal density and sentence-level agreement.
* An **Evidence Tab** displaying a per-sentence audit of quotes, negations, and filtered dictionary hits.
* **PDF & CSV Exporting** so you can archive your findings.

The app runs 100% locally. No API keys, no internet connection, no data ever leaves your machine.

---

## How it works

BiasLab utilizes a highly optimized four-stage processing pipeline spread across several modules to evaluate the context of the text deeply:

### Stage A: Context Building
Uses `NLTK` or regex fallbacks to tokenize text into discrete sentences. Analyzes each sentence to flag quoted spans, attribution verbs, and negations, packaging the result so the engine knows what is reporter voice versus quoted subjects.

### Stage B: Feature Extraction
Runs configured NLP lexicons against the sentences, actively suppressing hits that fall within quoted spans or near negations. Excluded phrase hits ensure that reporting on extreme topics isn't incorrectly flagged as biased reporting.

### Stage C: Algorithm & Scoring
Applies density conversions (occurrences per 1000 words) and smooth hyperbolic formulas alongside the sentence-level extraction data to produce the final 0-100 radar axis values.

### Stage D: Confidence Generation
Outputs explicit confidence scores and uncertainty zones ranging from `LOW` to `HIGH` by computing sentence agreement, evidence density, length, and quotation fractions. This tracks if a score is based on strong recurrent signals or a single passing remark.

---

## The Six Bias Dimensions

Each axis scores **0 (neutral) to 100 (extreme)**. Higher = more biased.

1. **Loaded Language:** Emotionally-charged vocabulary, exclamations, ALL-CAPS.
2. **Sentiment Imbalance:** Is the emotional tone lopsided positive or lopsided negative?
3. **Subjectivity:** Absolutes and opinion markers push this up; hedges pull it back down.
4. **Source Opacity:** Are sources named, or is it hiding behind "experts say" (weasel words)?
5. **Sensational Framing:** Dramatic verbs (slammed, lashed out) and superlative claims.
6. **Political Slant:** Raw intensity of Left-coded or Right-coded ideological vocabulary.

---

## Install and run in 60 seconds

You need Python 3.8 or newer. The application runs inside an isolated virtual environment (`venv`) safely away from your system Python.

### Windows (PowerShell)

```powershell
git clone https://github.com/notUbaid/BiasLab.git
cd BiasLab
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m biaslab
```

### macOS / Linux

```bash
git clone https://github.com/notUbaid/BiasLab.git
cd BiasLab
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m biaslab
```

### Running it again later

You only create the venv once. Every time after that:

```bash
# Windows
cd BiasLab
.\.venv\Scripts\Activate.ps1
python -m biaslab

# macOS / Linux
cd BiasLab
source .venv/bin/activate
python -m biaslab
```

---

## Limitations and Disclaimers

* **Language only.** It cannot verify facts or source reliability.
* **Sarcasm.** Context-aware quoting helps significantly, but deep irony and advanced sarcasm are still difficult to detect without LLM models. 
* **Zero Causality.** High scores reflect loaded *text content*, not necessarily the *author's innate intent*. 

---

## Credits

Built by **Ubaid**, with contributions from **Kush**.
