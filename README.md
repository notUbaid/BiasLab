<div align="center">

```
____  _           _          _
| __ )(_) __ _ ___| |    __ _| |__
|  _ \| |/ _` / __| |   / _` | '_ \
| |_) | | (_| \__ \ |__| (_| | |_) |
|____/|_|\__,_|___/_____\__,_|_.__/

         Media  Bias  Radar   ::   v2.0
```

### See the language patterns behind the news.

Paste an article, get a six-axis **Bias Radar**, flagged words, clickbait detection, confidence score, and plain-language tips — all in one local, zero-API Python app.

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Tkinter](https://img.shields.io/badge/GUI-Tkinter-FFC107?style=for-the-badge)
![Matplotlib](https://img.shields.io/badge/Charts-Matplotlib-11557c?style=for-the-badge)
![License](https://img.shields.io/badge/license-MIT-green?style=for-the-badge)
![Status](https://img.shields.io/badge/status-A%2B%2B%20demo%20ready-38bdf8?style=for-the-badge)

</div>

---

## Table of Contents

1. [What is this, explained to a 5-year-old](#what-is-this-explained-to-a-5-year-old)
2. [What is this, explained to a grown-up](#what-is-this-explained-to-a-grown-up)
3. [Screens you will see](#screens-you-will-see)
4. [Features at a glance](#features-at-a-glance)
5. [Install and run in 30 seconds](#install-and-run-in-30-seconds)
6. [How to operate the app](#how-to-operate-the-app)
7. [The six bias dimensions](#the-six-bias-dimensions)
8. [Extra signals (not on the radar)](#extra-signals-not-on-the-radar)
9. [Under the hood: the scoring math](#under-the-hood-the-scoring-math)
10. [Signal glossary](#signal-glossary)
11. [Tuning and customizing](#tuning-and-customizing)
12. [File structure](#file-structure)
13. [Worked example](#worked-example)
14. [FAQ](#faq)
15. [Limitations and honest disclaimers](#limitations-and-honest-disclaimers)
16. [Roadmap](#roadmap)
17. [Credits](#credits)

---

## What is this, explained to a 5-year-old

> Imagine a news article is a pizza.
>
> Sometimes a pizza has normal cheese and tomato. That's plain news.
>
> Sometimes a pizza has **way** too much salt, too many sprinkles, and a giant sign
> that says "BEST PIZZA IN THE WORLD, EAT IT NOW!!!". That pizza is trying to trick
> your tongue into thinking it's amazing.
>
> **BiasLab is a little robot that tastes the pizza for you.** It reads the article,
> notices all the salt and sprinkles and shouting, and draws you a picture showing
> exactly where the extra stuff is. Then it tells you, plainly:
>
> _"This is mostly normal pizza."_
> or
> _"This pizza is trying really hard to make you feel angry. Maybe go taste a
> different one before you decide."_

That's it. No ads, no tracking, no "AI magic." Just word-counting and simple math.

---

## What is this, explained to a grown-up

**BiasLab v2** is a **Python desktop app** that analyzes a news article and reports
how neutral its language is. You paste text, click **Analyze**, and you get:

* An **overall bias score** from 0 (reads like straight reporting) to 100 (reads
  like opinion/propaganda).
* A **six-axis Bias Radar** chart visualizing *where* the bias comes from.
* A **LEFT ↔ RIGHT political lean** number (-1.00 to +1.00) separate from the
  intensity score.
* A **Clickbait Gap** metric that compares headline drama to body drama.
* A **Confidence score** that tells you how much to trust the result given how
  long the article is.
* The **exact words and phrases** the analyzer flagged, grouped by axis.
* Plain-language **rewrite suggestions** tailored to the highest-scoring axes.
* Optional **CSV logging** so you can build a little personal dataset over time.

The app runs 100% locally. No API keys, no internet connection, no data ever leaves
your machine.

---

## Screens you will see

```
.--------------------------------------------------------------------------.
| BiasLab                                                                  |
| Media Bias Radar v2.0 - paste an article, see the hidden language patterns|
|--------------------------------------------------------------------------|
| 1. Paste an article                   | 2. Bias report                   |
|                                       | +-------+-------+--------+-----+ |
|  Headline / title                     | Overview| Radar | Flagged| Tips| |
|  [ Radical Politicians Slam...    ]   | +-------+-------+--------+-----+ |
|                                       |                                  |
|  Article body                         |  ARTICLE                         |
|  +---------------------------------+  |   Title       : Radical Polit... |
|  | In a truly shocking and         |  |   Analyzed on : 2026-04-20 ...   |
|  | devastating turn of events,     |  |   Word count  : 340              |
|  | the radical left lashed out...  |  |   Confidence  : 0.63 (HIGH)      |
|  |                                 |  |                                  |
|  +---------------------------------+  |  OVERALL                         |
|                                       |   Overall bias score :  51.0 /100|
|  [ Analyze ] [Load Sample] [Clear]    |   Verdict            : MODERATE  |
|                      [ Save to CSV ]  |   Political lean     : +0.64     |
|                                       |   Clickbait gap      :  81.7 /100|
|                                       |                                  |
|                                       |  AXIS BREAKDOWN                  |
|                                       |   Loaded Language      46.5 [###]|
|                                       |   Sentiment Imbalance   6.6 [#..]|
|                                       |   Subjectivity         42.0 [###]|
|                                       |   Source Opacity       81.7 [###]|
|                                       |   Sensational Framing  64.9 [###]|
|                                       |   Political Slant      64.3 [###]|
'--------------------------------------------------------------------------'
```

The **Radar Chart** tab shows a cyan polygon on a polar grid — one corner per
axis. A perfectly neutral article collapses to a point in the center; a heavily
biased one puffs out to the edges.

---

## Features at a glance

| Feature | What it does |
|---|---|
| **Six-dimension radar** | Visualizes bias across Loaded Language, Sentiment, Subjectivity, Sourcing, Framing, Political Slant. |
| **Smooth hyperbolic scoring** | Numbers spread naturally across 0–100 instead of saturating to binary 0/100. |
| **Multi-factor per axis** | Each axis combines 2–4 sub-signals (density, exclamations, ALL-CAPS, hedges, superlatives, quotes). |
| **Political lean indicator** | Separate LEFT ↔ RIGHT direction from -1.00 to +1.00. |
| **Clickbait Gap** | Detects headlines that are way more dramatic than their own article body. |
| **Confidence score** | Tells you how reliable the analysis is (short articles get a warning). |
| **Flagged word list** | See the exact phrases that drove each score. |
| **Rewrite suggestions** | Concrete tips keyed to whichever axis scored high. |
| **CSV session log** | Every run can be saved to `biaslab_sessions.csv` for later review. |
| **Built-in demo article** | One-click sample so you can show your teacher in 5 seconds. |
| **Dark themed tkinter UI** | Polished, presentation-ready, no external frameworks. |
| **100% local, no APIs** | Nothing leaves your machine. No keys. No login. No tracking. |

---

## Install and run in 30 seconds

**Requirements:** Python 3.8 or newer, and `matplotlib` for the radar chart.

```bash
# 1. Install matplotlib (tkinter ships with Python on Windows/macOS)
pip install matplotlib

# 2. Launch the app
python biaslab.py
```

That's it. A window will open. Click **Load Sample**, then **Analyze** to see it work.

> On some Linux distros, `tkinter` must be installed separately with
> `sudo apt install python3-tk`.

---

## How to operate the app

### Step 1 — paste the article

* **Headline / title** field: paste the article's title or headline. Optional, but
  it unlocks the Clickbait Gap metric.
* **Article body** box: paste the full article text. Minimum ~50 characters.

### Step 2 — click **Analyze**

The right side fills with a report across four tabs:

| Tab | What's in it |
|---|---|
| **Overview** | Headline, word count, confidence, overall score, verdict, political lean, clickbait gap, article stats, and a per-axis breakdown with its sub-factors. |
| **Radar Chart** | Matplotlib polar chart of the six bias axes. |
| **Flagged Words** | Which phrases triggered which axis, grouped and sorted. |
| **Suggestions** | Concrete rewrite tips chosen based on the highest-scoring axes. |

### Step 3 — optional actions

* **Load Sample** — fills the input fields with a purposely-biased article so you
  can demo the app instantly.
* **Clear** — wipes everything and resets the report panels.
* **Save to CSV** — appends the current analysis as one row to
  `biaslab_sessions.csv` in the project folder.

---

## The six bias dimensions

Each axis scores **0 (neutral) to 100 (extreme)**. Higher = more biased.

### 1. Loaded Language
Emotionally-charged vocabulary, exclamation marks, and ALL-CAPS shouting.
**Sub-signals:** sensational word density (60%), exclamation density (25%),
ALL-CAPS density (15%). Words like _shocking, devastating, bombshell, catastrophe,
scandal, meltdown_ trigger this axis.

### 2. Sentiment Imbalance
Is the emotional tone lopsided positive or lopsided negative?
**Formula:** `imbalance_ratio × confidence × 100`, where `imbalance_ratio =
|positive − negative| / total_sentiment_words` and `confidence` scales with how
many sentiment words appear per 1000 words.

### 3. Subjectivity
Absolutes and opinion markers push this up; hedges pull it back down by up to
30 points. Absolutes = _obviously, clearly, undeniably, make no mistake_. Hedges
= _may, might, could, appears, suggests, approximately_. Good journalism uses
hedges, so their presence reduces the subjectivity score.

### 4. Source Opacity
Is the article citing named sources or hiding behind "sources say"?
**Sub-signals:** weasel share (55%), raw weasel density (45%), quoted-speech
relief (up to -15 pts). Named citations _("according to John Doe", "in a
statement")_ are good; weasel phrasing _("sources say", "insiders claim",
"reportedly")_ is bad.

### 5. Sensational Framing
Dramatic verbs and us-vs-them language combined with superlative claims.
**Sub-signals:** framing word density (70%) + superlative density (30%).
Framing words: _slammed, lashed out, regime, mob, enemy, thugs_. Superlatives:
_worst, unprecedented, record-breaking, biggest ever_.

### 6. Political Slant
How ideologically loaded is the vocabulary overall? This axis reports *intensity*
(0–100). The *direction* is reported separately as Political Lean below.
Left-coded vocab: _progressive, marginalized, systemic, far-right, corporate greed_.
Right-coded vocab: _woke, liberal elite, socialist, patriot, open borders_.

---

## Extra signals (not on the radar)

### Political Lean  (−1.00 .. +1.00)
* −1.00 = all loaded words used are left-coded
* 0.00 = balanced or no loaded vocab
* +1.00 = all loaded words used are right-coded

The app translates this to plain English: _Strong/Moderate LEFT lean_,
_Roughly centered_, _Moderate/Strong RIGHT lean_.

### Clickbait Gap  (0 .. 100)
Compares the drama-density of the **headline** versus the drama-density of the
**body**. A score above ~40 means the headline is significantly more dramatic
than the story actually justifies — the classic clickbait pattern.

### Confidence  (0.0 .. 1.0)
Grows with article length using the same smooth curve used for scoring.
* ~100 words → 0.33 (LOW confidence, treat results as suggestive)
* ~300 words → 0.60 (MEDIUM)
* ~600 words → 0.75 (HIGH)
* ~1000 words → 0.83 (HIGH)

### Article stats
Raw counts surfaced in the Overview tab: paragraphs, exclamation marks, question
marks, ALL-CAPS words, and properly-quoted segments.

---

## Under the hood: the scoring math

Every axis follows the same three-step recipe:

1. **Count** signal phrases using regex word boundaries.
2. **Convert to density** per 1000 words so long and short articles are comparable.
3. **Smooth to 0–100** using a hyperbolic curve.

The smoothing function is the heart of v2:

```python
def smooth_score(value, half_point):
    if value <= 0:
        return 0.0
    return 100.0 * value / (value + half_point)
```

This curve gives:

| value (per 1000 words) | score |
|---|---|
| 0 | 0 |
| half_point | 50 |
| 3 × half_point | 75 |
| 9 × half_point | 90 |
| ∞ | approaches 100 but never reaches |

So a moderately biased article scores in the 40–60 range, a heavily biased one
in the 70–85 range, and a neutral one near 0 — **no more all-or-nothing
saturation**. The `half_point` is tuned per axis in the config section of
`biaslab.py`.

Sub-signals inside each axis are combined with a **weighted average** (weights
shown in the metric docstrings). Hedging words apply a **subtractive bonus**
that reduces Subjectivity. Quoted speech applies a **subtractive bonus** that
reduces Source Opacity.

The final overall score is the simple mean of the six axis scores.

---

## Signal glossary

<details>
<summary><b>Sensational words</b> (click to expand)</summary>

```
shocking, devastating, explosive, bombshell, outrageous, disgusting,
horrifying, terrifying, catastrophe, crisis, chaos, furious, enraged,
tragic, nightmare, meltdown, scandal, disaster, stunning, jaw-dropping,
unbelievable, unthinkable, alarming, apocalyptic, unprecedented, staggering,
horrendous, heartbreaking, gut-wrenching, ferocious, blistering, damning,
scathing, seething
```
</details>

<details>
<summary><b>Absolute / opinion markers</b></summary>

```
obviously, clearly, undeniably, without question, everyone knows,
nobody denies, of course, naturally, evidently, surely, certainly,
undoubtedly, definitely, plainly, absolutely, needless to say,
make no mistake, beyond doubt, anyone can see, the fact is,
the simple truth
```
</details>

<details>
<summary><b>Hedge markers (good — reduces Subjectivity)</b></summary>

```
may, might, could, appears, seems, likely, possibly, perhaps, apparently,
suggests, indicates, roughly, approximately, around, nearly, in part,
to some extent, is believed to, estimates, preliminary, tentative
```
</details>

<details>
<summary><b>Positive sentiment words</b></summary>

```
success, triumph, victory, breakthrough, hope, thriving, brilliant,
excellent, wonderful, great, positive, progress, achievement, innovative,
inspiring, praise, celebrated, landmark, historic, milestone, boost,
soar, surge, rally, gains, welcomed, applauded
```
</details>

<details>
<summary><b>Negative sentiment words</b></summary>

```
failure, defeat, plunge, crumble, broken, dangerous, threat, fear,
worry, concern, worst, terrible, awful, weak, damage, harmed, hurt,
collapse, ruined, criticized, condemned, slump, fell, declined,
plummeted, blasted, rejected
```
</details>

<details>
<summary><b>Left-coded vocabulary</b></summary>

```
progressive, equity, marginalized, systemic, far-right,
white supremacist, xenophobic, climate crisis, reproductive rights,
wealth gap, corporate greed, anti-worker, science denier, income inequality,
living wage, union-busting, disenfranchised, structural racism,
late-stage capitalism, dog whistle
```
</details>

<details>
<summary><b>Right-coded vocabulary</b></summary>

```
woke, liberal elite, mainstream media, illegal aliens, radical left,
socialist, communist, globalist, patriot, traditional values,
law and order, entitlement, welfare state, big government, open borders,
cancel culture, job creator, family values, silent majority,
career politician, deep state
```
</details>

<details>
<summary><b>Framing words (us-vs-them)</b></summary>

```
slammed, attacked, blasted, ripped, lashed out, tore into, unloaded on,
savaged, pounced, enemy, traitor, hero, coward, regime, crackdown, mob,
radicals, thugs, puppet, warmonger, witch hunt, kangaroo court, clash,
showdown, stormed
```
</details>

<details>
<summary><b>Superlatives</b></summary>

```
worst, best, greatest, biggest, largest, smallest, unprecedented, historic,
record-breaking, first-ever, never before, all-time low, all-time high,
biggest ever
```
</details>

<details>
<summary><b>Citation signals (good)</b></summary>

```
according to, said, stated, reported, noted, announced, confirmed,
told reporters, press release, in a statement, testified, wrote in,
explained, acknowledged, admitted, documents show, court records
```
</details>

<details>
<summary><b>Weasel signals (bad)</b></summary>

```
sources say, experts say, critics claim, it's rumored, many believe,
some argue, reportedly, allegedly, it is believed, people are saying,
insiders say, observers note, anonymous sources, familiar with the matter,
industry watchers, some say
```
</details>

---

## Tuning and customizing

All tuning knobs are at the top of `biaslab.py`, in **Section 2**. You can safely
edit these without touching any logic:

```python
# Half-points: the density (per 1000 words) at which an axis scores 50.
HALF_SENSATIONAL    = 12.0
HALF_ABSOLUTE       =  8.0
HALF_FRAMING        = 12.0
HALF_SLANT          = 18.0
HALF_WEASEL         =  6.0
HALF_EXCLAIM        =  5.0
HALF_CAPS           =  3.0
HALF_SUPERLATIVE    =  6.0
HALF_CLICKBAIT_GAP  = 40.0
HALF_CONFIDENCE     = 200.0
```

* **Smaller number** → the axis is more sensitive, a little signal moves the
  score a lot.
* **Bigger number** → the axis is less sensitive, signal has to be really
  strong to move the needle.

You can also extend the lexicons directly. For example, to teach the analyzer
about a new dramatic buzzword, just append it to `SENSATIONAL_WORDS`:

```python
SENSATIONAL_WORDS.append("mind-blowing")
```

The changes take effect the next time you run the app.

---

## File structure

```
BiasLab/
├── biaslab.py                 <- the whole application (one file, heavily commented)
├── biaslab_sessions.csv       <- written at runtime when you click "Save to CSV"
├── README.md                  <- you are here
└── .gitignore
```

Inside `biaslab.py` the 10 sections are:

| # | Section | What it does |
|---|---|---|
| 1 | Imports | csv, re, tkinter, matplotlib, datetime, math |
| 2 | Configuration | Lexicons, half-points, radar labels, verdict thresholds |
| 3 | Text helpers | normalize, word count, phrase counting, smooth_score, stats |
| 4 | Metric functions | Six scoring functions + clickbait_gap + confidence |
| 5 | Aggregator | `analyze_article()` — runs everything, returns one dict |
| 6 | Verdict & tips | Label, lean description, suggestion builder |
| 7 | CSV logging | Ensure-exists + append-one-row |
| 8 | Tkinter GUI | `BiasLabApp` class — window, tabs, handlers, radar draw |
| 9 | Demo article | Long, mixed-signal sample for instant class demos |
| 10 | main() | Starts the Tk event loop |

---

## Worked example

Here's what the built-in demo article produces:

```
ARTICLE
  Title       : Radical Politicians Slam Patriotic Plan In Shocking Meltdown...
  Analyzed on : 2026-04-20 12:34:56
  Word count  : 340
  Confidence  : 0.63 (HIGH)

OVERALL
  Overall bias score :  51.0 / 100
  Verdict            : MODERATE BIAS
    -> Notable slant - read a second source to balance it.

  Political lean     : +0.64  (Strong RIGHT lean)
  Clickbait gap      :  81.7 / 100  (title drama 272.73, body drama 94.22)

ARTICLE STATS
  Paragraphs       : 6
  Exclamations     : 0
  Questions        : 0
  ALL-CAPS words   : 0
  Quoted segments  : 0

AXIS BREAKDOWN
  Loaded Language         46.5   [##########............]
      - sensational density per 1000: 41.18
      - exclamation density per 1000:  0.00
      - all caps density per 1000:     0.00

  Sentiment Imbalance      6.6   [#.....................]
      - positive hits:     2
      - negative hits:     3
      - imbalance ratio:   0.20
      - sentiment density: 14.71
      - confidence:        0.33

  Subjectivity            42.0   [#########.............]
      - absolute density:     23.53
      - hedge density:         2.94
      - hedge relief points:   8.82

  Source Opacity          81.7   [##################....]
      - named citations:      2
      - weasel citations:     8
      - quoted segments:      0
      - quote relief points:  0.00

  Sensational Framing     64.9   [##############........]
      - framing density:       26.47
      - superlative density:   14.71

  Political Slant         64.3   [##############........]
      - left hits:    1
      - right hits:   7
      - loaded density: 23.53
```

Every number is explainable.

---

## FAQ

<details>
<summary><b>Is this real AI / machine learning?</b></summary>

No, and that's on purpose. BiasLab uses curated lexicons plus simple math
(counting, density, a hyperbolic smoothing curve, weighted averages). This
keeps the code small, auditable, and easy to defend in a class presentation.
A machine-learning classifier could be swapped in later as a drop-in
replacement for the six scoring functions.
</details>

<details>
<summary><b>Does BiasLab tell me if an article is true?</b></summary>

No. BiasLab measures **language patterns**, not facts. An article can score
0 on every axis and still be factually wrong; an opinion piece can score
high and still be accurate. Treat BiasLab as a *signal*, not a verdict on
truth.
</details>

<details>
<summary><b>Is my data being sent anywhere?</b></summary>

No. Everything runs locally in Python. The only file written is
`biaslab_sessions.csv`, and only when you explicitly click **Save to CSV**.
</details>

<details>
<summary><b>Why do my axis scores look different than a classmate's?</b></summary>

Because the scores depend on the article you paste. Two different articles
will produce two different radars. If you paste the same article, the
results will be identical every time — the analyzer is fully deterministic.
</details>

<details>
<summary><b>Can it handle non-English articles?</b></summary>

Partially. The math works on any text, but the lexicons are English-only,
so signals won't fire on, say, a Spanish article. Extending to another
language is as easy as translating each lexicon list.
</details>

<details>
<summary><b>Why does a neutral short piece sometimes score non-zero?</b></summary>

Because one or two sentiment words in a very short article can spike the
per-1000-word density. This is exactly why the app reports a **Confidence**
score — short articles get LOW confidence, which means "treat these numbers
as suggestive, not conclusive."
</details>

<details>
<summary><b>How long can the article be?</b></summary>

There is no upper limit. Tkinter's text widget handles very long articles
fine. The scoring is density-based, so a 5000-word article is handled
exactly like a 500-word one — just with higher confidence.
</details>

<details>
<summary><b>Can I add my own words to the lexicons?</b></summary>

Yes. Open `biaslab.py`, find the lexicon list in Section 2, and append your
phrase. Multi-word phrases (like `"according to"`) work exactly the same as
single words because regex word boundaries handle both. No other changes
are needed — save the file and rerun.
</details>

---

## Limitations and honest disclaimers

* **Language-only.** BiasLab cannot verify facts, sources, or images.
* **Bag-of-phrases.** The analyzer doesn't understand context. Sarcasm, quoting
  biased phrases to criticize them, and sophisticated rhetoric can fool it.
* **Curated lexicons.** The word lists reflect their authors' judgment. They
  are a starting point, not a final ground truth.
* **English only.**
* **No causality claims.** A high score doesn't mean the *author* is biased —
  only that the *language* is loaded.

Use BiasLab as a **magnifying glass**, not a **judge**.

---

## Roadmap

Planned improvements (good ideas for v3):

- [ ] Per-paragraph heatmap: highlight the article text itself with colored
      underlines for each flagged phrase.
- [ ] Side-by-side comparison: paste two articles about the same story and
      compare their radars.
- [ ] Export report as PDF or PNG.
- [ ] Pluggable language packs (`biaslab_es.py`, `biaslab_fr.py`).
- [ ] A small ML layer on top (still optional) for sarcasm and irony handling.
- [ ] URL fetcher: paste a link, BiasLab pulls the text automatically.

---

## Credits

* Built by **Ubaid** as a class project, version 2.0.
* The six-axis taxonomy is inspired by media-literacy research from
  organizations such as AllSides, Ad Fontes Media, and the News Literacy
  Project, adapted and simplified for educational use.
* The smoothing curve is the classic Michaelis-Menten shape
  (`V / (V + K)`), repurposed from chemistry to do nice bias scoring.

---

<div align="center">

_If BiasLab helped you read the news a little smarter, that's the whole point._

</div>
