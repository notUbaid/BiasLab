# BiasLab v3: Context-Aware Media Bias Radar
**10-Minute Video Presentation Script & Technical Outline**

---

## Part 1: Introduction & The Problem (0:00 - 2:00)

**[Visual Idea: Start with the BiasLab GUI open on screen. Keep a browser window nearby with a news article.]**

**"Welcome to BiasLab."**
* **The Pitch:** We are surrounded by media narratives, making it hard to separate objective reporting from sensational framing and subjective slant. BiasLab is a Python text-analysis tool that acts like an X-Ray for articles. Simply provide an article, and it highlights the *words and sentence structures* that indicate bias.
* **Demo:** *[Copy an article into the app and hit Analyze]*
  * Observe the **Multi-Dimensional Bias Radar**: It plots bias across 6 distinct NLP vectors (Loaded Language, Sentiment, Subjectivity, Sourcing, Framing, Slant). 
  * Highlight the **Overall Score** and **Confidence Zone**. 
  * Transition to the **Evidence Tab**: Emphasize how every single score is fully deterministic and backed by transparent, sentence-by-sentence analytical evidence.

**The Challenge Behind the Scenes:**
* Standard analysis tools are often just "dumb word counters." They flag words like "terrible" or "disastrous" without looking at the context.
* **The Flaw:** If a completely neutral reporter writes: *The President said, "This is a terrible disaster"*, older approaches severely penalize the article. They get confused by **quotes**, **negations** ("not terrible"), and **attributions** ("he claimed").
* **The Solution:** We built BiasLab as a smart, organized Python app. BiasLab uses the dependable `NLTK` library along with clever rules to understand *sentence structure*, filtering out false alarms entirely.

---

## Part 2: Architectural Overview (2:00 - 3:00)

**[Visual Idea: Show a simple block diagram or file tree of the `biaslab/` directory on screen]**

* **Rules-Based vs. AI:** Why avoid AI like ChatGPT? AI can be unpredictable, requires internet access, and can be slow. BiasLab relies on clear rules, direct evaluation, and balanced scoring math. Every single decimal point on the radar is fully clear and easy to explain.
* **The 4-Stage Pipeline:** Processing happens step by step: Reading the Context, Finding the Features, Generating the Score, and Creating Insights.

Let's dissect the repository and see how each module coordinates.

---

## Part 3: Deep Dive into the Codebase (3:00 - 8:30)

**[Visual Idea: Open up an IDE (VSCode/PyCharm) and walk through the actual files one by one]**

### 1. `config.py` (The Dictionaries)
* **What it does:** This file holds all our word lists and settings.
* **How it works:** It stores groups of specific trigger words and phrases. Examples include `SENSATIONAL_WORDS` ("bombshell") or `WEASEL_SIGNALS` ("experts say"). 
* **Why it matters:** This acts as the foundation of our tool. BiasLab makes choices directly from these clear, easy-to-update lists.

### 2. `preprocessing.py` (Stage A: Context Builder)
* **What it does:** Turns messy text into organized sentences. We use `NLTK` to accurately break the text down into analyzable chunks.
* **How it works:** Before any scoring happens, the code scans the area to understand what's there:
  * **Finding Quotes:** Pinpointing exactly where someone is being quoted.
  * **Finding Attributions:** Identifying words ("stated", "warned") that tell us who said what.
  * **Finding Negations:** Spotting words like "not" or "never" to correctly flip the meaning of following words.
* **The Result:** The next steps in our tool receive parts of text rich with context, rather than just simple words, allowing a smarter analysis.

### 3. `features.py` (Stage B: Finding the Features)
* **What it does:** Checks the sentences against the word lists in `config.py` while keeping an eye out for **Context Rules**.
* **How it works:** 
  * The author's own words get a standard weight of `1.0`.
  * If a biased word is found inside a quote, it correctly gets dropped to a much lower weight of `0.15` because quoting someone else doesn't make the reporter biased.
  * If the word "bad" appears, but right after a "not", the score drops significantly.
* **The Breakthrough:** This careful tracking gets rid of the false alarms that usually break simple analysis tools.

### 4. `scorer.py` & `confidence.py` (Stage C & D: The Algorithm)
* **What it does:** Turns rough point values into scores that fit neatly on a 0-100 radar chart.
* **How it works (`scorer.py`):** 
  * Analysis measures scores **per 1000 words**, to keep it fair whether an article is long or short.
  * We use a special **Smoothing Formula** which ensures the highest score never goes completely past `100`, keeping things grounded even if an article is highly emotional.
* **How it works (`confidence.py`):**
  * It acts as a safety checker. It looks at the article's length, the amount of evidence found, and how many quotes are there to give a **Confidence Zone** (`LOW`, `MEDIUM`, or `HIGH`). We don't just give answers; we measure how reliable they are.

### 5. `explain.py`, `export.py`, and `gui.py` (The Output)
* **`explain.py`:** Takes the number outputs and offers real, easy-to-read suggestions (e.g. *"Replace dramatic words with plain facts"*).
* **`export.py`:** Saves your exact results to a clean PDF report or an Excel spreadsheet for you to keep. 
* **`gui.py`:** Brings all the code together to create the app window that you saw at the beginning.

---

## Part 4: Conclusion (8:30 - 10:00)

**[Visual Idea: Cut back to the GUI, specifically showcasing the precision of the Evidence Tab]**

**Summary:**
* BiasLab goes far beyond simple word counting. It is a powerful, organized Python tool that uses sentence breaking, quote checking, and balanced scoring to provide deep, context-aware analysis.
* **Key Takeaway:** AI chatbots are fun, but deep analytical tasks need **clear, steady rules**. BiasLab relies on transparent lists and clear math, meaning it can explain exactly where every score came from. There's no guesswork or magic, just true transparency.

**Call to Action:**
* The project's structure makes it very easy to work with. Adding new bias categories or changing the scoring is completely separate and as simple as modifying `config.py`.

*(End Video)*
