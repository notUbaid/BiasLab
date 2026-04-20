# BiasLab v3: Context-Aware Media Bias Radar
**10-Minute Video Presentation Script & Technical Outline**

---

## Part 1: Introduction & The Problem (0:00 - 2:00)

**[Visual Idea: Start with the BiasLab GUI open on screen. Keep a browser window nearby with a news article.]**

**"Welcome to BiasLab."**
* **The Pitch:** We are inundated with media narratives, making it challenging to isolate objective reporting from sensational framing and subjective slant. BiasLab is an advanced, fully localized Python NLP engine that acts as a diagnostic X-Ray for text. Simply provide an article, and it visualizes the *underlying lexical heuristics and structural patterns* that indicate bias.
* **Demo:** *[Copy an article into the app and hit Analyze]*
  * Observe the **Multi-Dimensional Bias Radar**: It plots bias across 6 distinct NLP vectors (Loaded Language, Sentiment, Subjectivity, Sourcing, Framing, Slant). 
  * Highlight the **Overall Score** and **Confidence Zone**. 
  * Transition to the **Evidence Tab**: Emphasize how every single score is fully deterministic and backed by transparent, sentence-by-sentence analytical evidence.

**The Architecture Challenge:**
* Standard analysis tools are effectively "naive bag-of-words classifiers." They indiscriminately flag words like "terrible" or "disastrous."
* **The Flaw:** If a completely neutral reporter writes: *The President said, "This is a terrible disaster"*, legacy approaches severely penalize the article. They fail on **quotes**, **negation scopes** ("not terrible"), and **attribution clauses** ("he claimed").
* **The Solution:** We architected BiasLab into a sleek, modular Python package. BiasLab utilizes the robust `NLTK` library coupled with regex-driven algorithmic contexts to understand *syntactic boundaries*, completely eliminating naive false positives. 

---

## Part 2: Architectural Overview (2:00 - 3:00)

**[Visual Idea: Show a simple block diagram or file tree of the `biaslab/` directory on screen]**

* **Deterministic vs. Generative:** Why avoid LLMs (like ChatGPT)? LLMs are non-deterministic black boxes prone to hallucination, requiring API dependencies and high inference costs. BiasLab relies on strict linguistic taxonomies, deterministic evaluation, and hyperbolic smoothing mathematics. Every single decimal point on the radar is mathematically justifiable and fully explicable.
* **The 4-Stage NLP Pipeline:** Processing occurs sequentially via a 4-tier pipeline: Context Preprocessing, Feature Matrix Extraction, Hyperbolic Scoring, and Insights Generation. 

Let's dissect the repository and see how each module coordinates.

---

## Part 3: Deep Dive into the Codebase (3:00 - 8:30)

**[Visual Idea: Open up an IDE (VSCode/PyCharm) and walk through the actual files one by one]**

### 1. `config.py` (The Lexical Taxonomies)
* **What it does:** This module houses our constant schemas and hyperparameter tuning dials. 
* **How it works:** It stores categorized embeddings of indicator phrases. Examples include `SENSATIONAL_WORDS` ("bombshell") or `WEASEL_SIGNALS` ("experts say"). 
* **Why it matters:** This acts as the foundational explanation matrix. Our classifications are drawn directly from these rigid, tunable linguistic mappings.

### 2. `preprocessing.py` (Stage A: Context Builder)
* **What it does:** Transforms raw unstructured text into structured, manageable arrays. We utilize `NLTK` to accurately segment text down to an `AnnotatedSentence` dataclass construct.
* **How it works:** Before any scoring is executed, sophisticated boundary scans map the environment:
  * **Quote Segmentation:** Precisely indexing spans representing direct discourse.
  * **Attribution Resolution:** Identifying speech-act verbs ("stated", "warned") that project a subject's statement.
  * **Negation Scopes:** Locating structural negation ("not", "never") and identifying its blast radius.
* **The Result:** Downstream extractors are fed context-rich nodes rather than raw strings, enabling fully context-aware NLP tokenization.

### 3. `features.py` (Stage B: Feature Matrix Extraction)
* **What it does:** Maps sentences against the `config.py` lexicons while applying localized **Context Modifiers**.
* **How it works:** 
  * Intrinsic Author Voice receives a nominal weight of `1.0`.
  * Matches occurring within `preprocessing.py`'s quoted spans are violently discounted by an `0.15` coefficient because a reporter quoting someone else is not bias natively.
  * If the word "bad" surfaces, but is intercepted by a preceding negation scope, the hit is discounted to a `0.20` coefficient.
* **The Engineering Breakthrough:** This pipeline mathematically nullifies the false-positive limitations typical of standard NLP sentiment packages.

### 4. `scorer.py` & `confidence.py` (Stage C & D: The Algorithm)
* **What it does:** Normalizes raw extraction matrices into standardized 0-100 radar vectors.
* **How it works (`scorer.py`):** 
  * Analysis rests on **Density per 1000 words**, thereby neutralizing length bias between articles. 
  * We process scores through a **Hyperbolic Smoothing Curve** `Score = 100 * (val / (val + K))`. This ensures asymptotic normalization mapping gracefully to `100`, eliminating mathematical saturation breaking the scale on highly emotional pieces.
* **How it works (`confidence.py`):**
  * It employs a sophisticated Multi-Signal Assessment involving article length vector dimensions, evidence density bounds, and quote-saturation limits to output an explicit **Uncertainty Zone** (`LOW`, `MEDIUM`, `HIGH`). We don't just output data; we measure data reliability.

### 5. `explain.py`, `export.py`, and `gui.py` (The Output)
* **`explain.py`:** Takes the raw math metrics and generates English suggestions (e.g. *"Replace dramatic words with plain facts"*).
* **`export.py`:** Uses `matplotlib` and `csv` to dump the exact session to a professional PDF report or logging spreadsheet. 
* **`gui.py`:** Uses `Tkinter` and `matplotlib.backends.tkagg` to project these Python models into the standalone desktop interface you saw at the beginning.

---

## Part 4: Conclusion (8:30 - 10:00)

**[Visual Idea: Cut back to the GUI, specifically showcasing the precision of the Evidence Tab]**

**Architectural Summary:**
* BiasLab transcends naive word counting. It is a highly optimized, modular Python engine leveraging sentence tokenization, quote mapping, and hyperbolic density smoothing to execute rigorous contextual NLP analysis.
* **Key Takeaway:** Generative AI solutions are impressive, but deeply analytical problem sets require **transparent determinism**. Because BiasLab strictly enforces linguistic taxonomies mapped to geometric math curves, it can explicitly justify every single axis on its radar. We eliminate black boxes and offer surgical transparency.

**Call to Action:**
* The architecture provides an exceptionally manageable surface. Introducing new classification schema or adapting the hyperbolic math is an entirely decoupled process via `config.py`.

*(End Video)*
