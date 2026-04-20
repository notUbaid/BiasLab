<div align="center">

```
____  _           _          _
| __ )(_) __ _ ___| |    __ _| |__
|  _ \| |/ _` / __| |   / _` | '_ \
| |_) | | (_| \__ \ |__| (_| | |_) |
|____/|_|\__,_|___/_____\__,_|_.__/

         Media  Bias  Radar   ::   v3.0
```

### Context-Aware News Analysis System.

Paste an article, get a six-axis **Bias Radar**, sentence-level evidence summaries, PDF reporting, and uncertainty tracking. BiasLab v3 transcends simple word counting by analyzing sentences in context — properly ignoring quoted speech and accounting for negations to eliminate false positives. Runs 100% locally.

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Tkinter](https://img.shields.io/badge/GUI-Tkinter-FFC107?style=for-the-badge)
![NLTK](https://img.shields.io/badge/NLP-NLTK-2ca02c?style=for-the-badge)
![License](https://img.shields.io/badge/license-MIT-green?style=for-the-badge)

</div>

---

## What's New in v3.0

The v3 architecture transforms the 2,000-line monolith of v2 into a sleek, modular Python package. But the biggest upgrade is the new **context-aware pipeline**:

* **False-Positive Elimination:** BiasLab now splits text into an `AnnotatedSentence` structure using NLTK.
* **Quote Immunity:** Heavy bias words inside direct quotes are correctly flagged as *quotes* rather than *authorial bias*, reducing false alarms on neutral articles covering extreme subjects.
* **Negation Awareness:** The engine detects negations ("not", "never") preceding a flagged word and downweights them.
* **Attribution Filtering:** Sentences using attribution verbs ("he claimed", "she warned") are assigned lower confidence weights than the author's naked voice.
* **New Unified Evidence UI:** Instead of just a list of words, you get a full per-sentence breakdown explaining *why* the score was generated.
* **PDF Export Generation:** Export your session directly to a professional PDF report.

---

## What is this?

**BiasLab v3** is a Python desktop app that analyzes a news article and measures how neutral its language is. You paste text, click **Analyze**, and you get:

* An **overall bias score** from 0 (reads like straight reporting) to 100 (reads like opinion/propaganda).
* A **six-axis Bias Radar** chart visualizing *where* the bias comes from.
* A **LEFT <-> RIGHT political lean** number (-1.00 to +1.00) separate from the intensity score.
* A **Confidence score & Uncertainty Zone** that explicitly tells you how reliable the score is based on signal density and sentence-level agreement.
* An **Evidence Tab** displaying a per-sentence audit of quotes, negations, and filtered dictionary hits.
* **PDF & CSV Exporting** so you can archive your findings.

The app runs 100% locally. No API keys, no internet connection, no data ever leaves your machine.

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

## How to operate the app

### Step 1 - Paste the article
* **Headline / title** field: Paste the article's title. This unlocks the Clickbait Gap metric.
* **Article body** box: Paste the full article text. Minimum ~50 characters.

### Step 2 - Click **Analyze**
The right side fills with a report across four tabs:

| Tab | What's in it |
|---|---|
| **Summary** | Overall score, verdict, graphical score callout. |
| **Radar** | Matplotlib polar chart of the six bias axes. |
| **Evidence** | Per-sentence audit log detailing quotes, attributions, negations, and extracted lexicon hits. |
| **Tips** | Concrete rewrite tips generated from the underlying signal metrics. |

### Step 3 - Export (Optional)
* **Export PDF** - Generates a professional 1-page report detailing the radar and overall breakdown.
* **Save CSV** - Appends your run to a local `biaslab_sessions.csv` file.

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

## Under the Hood: v3 Architecture

BiasLab v3 utilizes a highly optimized four-stage processing pipeline spread across several modules:

### Stage A: Preprocessing (`preprocessing.py`)
Uses NLTK or regex fallbacks to tokenize text into discrete sentences. Analyzes each sentence to flag quoted spans, attribution verbs, and negations, packaging the result into an `AnnotatedSentence` dataclass.

### Stage B: Feature Extraction (`features.py`)
Runs the lexicons against the sentences, actively suppressing hits that fall within quoted spans or near negations. Excluded phrase hits are recorded in the `excluded_quoted` and `negated_phrases` metrics.

### Stage C: Scoring (`scorer.py`)
Applies density conversions and smooth hyperbolic formulas alongside the sentence-level extraction data to produce the final 0-100 values.

### Stage D: Confidence (`confidence.py`)
Outputs explicit confidence scores and uncertainty zones ranging from `LOW` to `HIGH` by computing sentence agreement, evidence density, length, and quotation fractions.

---

## Package Structure

BiasLab has been rewritten into a proper Python package:

```
BiasLab/
├── biaslab/
│   ├── __init__.py
│   ├── __main__.py          <- Entry point (`python -m biaslab`)
│   ├── config.py            <- Lexicon arrays, weighting constants, thresholds
│   ├── preprocessing.py     <- Sentence tokenization & context flags
│   ├── features.py          <- Context-aware phrase matching
│   ├── scorer.py            <- Mathematical aggregations and radar calculation
│   ├── confidence.py        <- Uncertainty algorithms 
│   ├── explain.py           <- String generation and UI tips
│   ├── export.py            <- PDF & CSV modules
│   └── gui.py               <- Tkinter application 
├── requirements.txt
└── README.md
```

You can customize the lexicons or tune the severity curves directly by editing `biaslab/config.py`.

---

## Limitations and Disclaimers

* **Language only.** It cannot verify facts or source reliability.
* **Sarcasm.** Context-aware quoting helps significantly, but deep irony and advanced sarcasm are still difficult to detect without LLM models. 
* **Zero Causality.** High scores reflect loaded *text content*, not necessarily the *author's innate intent*. 

---

## Credits

Built by **Ubaid**, version 3.0.
Refactored to context-aware modular architecture for production deployments.
</div>
