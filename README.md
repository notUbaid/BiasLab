# BiasLab – Adaptive Decision Intelligence

## Project Overview

BiasLab is a Python GUI app for analyzing **real-life decisions** (small to major) and identifying whether your current leaning is:

- practically justified, or
- being distorted by pressure and bias.

It is designed for college-level Python with basic-to-moderate concepts while still giving deep, practical output.

## How BiasLab Works

BiasLab adapts its questions based on your decision text and your previous answers. It only asks what is necessary, and then explains what is likely going on in plain language.

BiasLab evaluates two options in parallel:

1. **Option strength** (usefulness, long-term fit, evidence, compatibility, tradeoff)
2. **Pressure signals** (emotion, urgency, social pressure, sunk cost, identity attachment, loss aversion, novelty pull)

It then combines these with fairness and foresight checks to compute:

- Bias risk
- Clarity score
- Gap between options

### Important Practical Rule

A preference is **not** flagged as unfair bias if there is strong practical support.

Example: choosing iPhone because of ecosystem compatibility with your existing Apple devices can be treated as a valid practical reason when evidence and need-fit are also strong.

## Features

- Adaptive, one-question-at-a-time wizard
- Works for purchase, relationship, career, finance, academic, health, social, and generic dilemmas
- Auto-detects options if you write “X or Y” in the decision
- Bias-type detection with simple action steps
- Plain-language report and decision radar
- Session logging to `biaslab_sessions.csv`

## Technologies Used

- Python 3
- Tkinter
- Matplotlib
- NumPy
- CSV / Datetime / OS modules

## How to Run

1. Install dependencies:

	```bash
	pip install matplotlib numpy
	```

2. Run the app:

	```bash
	python biaslab.py
	```

3. Enter your decision, define Option A and Option B (or let BiasLab infer them), and answer the prompts one by one.

## Output

- In-app decision report with bias types and next steps
- Radar chart of plain-language pressure signals
- `biaslab_sessions.csv` with each session's metrics

## Privacy and Safety

- BiasLab saves sessions locally to `biaslab_sessions.csv`.
- It does not send any data anywhere.
- This tool is for decision support only, not professional medical/legal/financial advice.

## Academic Purpose

This project demonstrates applied decision modeling with psychology-informed heuristics, weighted scoring, and practical explainability using foundational Python concepts.



