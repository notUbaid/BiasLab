import csv
import datetime
import os
import re
import tkinter as tk
from tkinter import ttk

import matplotlib.pyplot as plt
import numpy as np


# ==========================================================
# SECTION 1: SHARED CONFIGURATION (KEYS, WEIGHTS, LABELS)
# ==========================================================

CONTEXT_KEYWORDS = {
    "purchase": [
        "buy",
        "price",
        "cost",
        "deal",
        "discount",
        "sale",
        "order",
        "purchase",
        "shop",
        "shopping",
        "product",
        "item",
        "phone",
        "laptop",
        "tablet",
        "headphone",
        "earbuds",
        "camera",
        "car",
        "bike",
        "scooter",
        "house",
        "apartment",
        "rent",
    ],
    "relationship": [
        "date",
        "dating",
        "relationship",
        "partner",
        "boyfriend",
        "girlfriend",
        "crush",
        "love",
        "like",
        "approach",
        "confess",
        "ask out",
        "proposal",
        "marry",
        "marriage",
        "breakup",
        "divorce",
        "commit",
        "commitment",
        "situationship",
    ],
    "career": [
        "job",
        "career",
        "work",
        "internship",
        "promotion",
        "resign",
        "quit",
        "switch",
        "role",
        "company",
        "startup",
        "business",
        "degree",
        "masters",
        "mba",
    ],
    "finance": [
        "money",
        "budget",
        "save",
        "saving",
        "invest",
        "investment",
        "stock",
        "crypto",
        "fund",
        "loan",
        "emi",
        "rent",
        "debt",
        "credit",
        "interest",
    ],
    "academic": [
        "exam",
        "study",
        "college",
        "school",
        "course",
        "major",
        "minor",
        "university",
        "gpa",
        "assignment",
        "project",
        "thesis",
        "research",
    ],
    "health": [
        "health",
        "diet",
        "sleep",
        "workout",
        "exercise",
        "gym",
        "doctor",
        "therapy",
        "mental",
        "stress",
        "anxiety",
        "medicine",
        "treatment",
    ],
    "social": [
        "friend",
        "friends",
        "party",
        "group",
        "hangout",
        "meet",
        "text",
        "call",
        "message",
        "event",
        "invite",
        "invite",
        "social",
    ],
}

MAJOR_DECISION_KEYWORDS = [
    "marry",
    "marriage",
    "breakup",
    "divorce",
    "career",
    "job",
    "degree",
    "business",
    "house",
    "loan",
    "move",
    "relocate",
    "commit",
    "commitment",
    "investment",
    "proposal",
    "mortgage",
    "visa",
    "abroad",
    "relocation",
    "surgery",
    "diagnosis",
    "treatment",
    "insurance",
]

SMALL_DECISION_KEYWORDS = [
    "ice cream",
    "snack",
    "food",
    "drink",
    "movie",
    "shirt",
    "weekend",
    "today",
    "tonight",
    "coffee",
    "tea",
    "dessert",
    "game",
    "music",
    "playlist",
    "meme",
    "post",
    "reply",
    "text back",
    "call back",
    "outfit",
    "order food",
]

OPTION_CRITERIA_KEYS = ["need_fit", "long_term", "tradeoff", "evidence", "compatibility"]

RATIONAL_WEIGHTS = {
    "evidence": 0.35,
    "need_fit": 0.20,
    "long_term": 0.20,
    "compatibility": 0.15,
    "tradeoff": 0.10,
}

DISTORTION_WEIGHTS = {
    "bias_pressure": 0.40,
    "foresight_gap": 0.20,
    "fairness_risk": 0.15,
    "weak_choice_penalty": 0.15,
    "low_evidence_penalty": 0.10,
}

BIAS_PRESSURE_KEYS = [
    "emotion",
    "urgency",
    "social_pressure",
    "sunk_cost",
    "identity_attachment",
    "loss_aversion",
    "novelty_pull",
]


def normalize(value):
    return max(0.0, min(1.0, value / 10.0))


def clamp01(value):
    return max(0.0, min(1.0, value))


def classify_risk(risk_score):
    if risk_score < 0.30:
        return "High Decision Integrity"
    if risk_score < 0.60:
        return "Balanced but Needs Reflection"
    return "Elevated Distortion Risk"


class BiasLab:
    """BiasLab app with clearly separated UI, scoring logic, and report text logic."""

    # ======================================================
    # SECTION 2: APP STATE + APP BOOTSTRAP
    # ======================================================

    def __init__(self, root):
        self.root = root
        self.root.title("BiasLab - Practical Decision Intelligence")
        self.root.geometry("1120x860")
        self.root.configure(bg="#f3f5fb")

        self.answers = {}
        self.signal_map = {}
        self.option_a_sliders = {}
        self.option_b_sliders = {}

        self.decision_text = None
        self.option_a_var = tk.StringVar(value="Option A")
        self.option_b_var = tk.StringVar(value="Option B")
        self.leaning_var = tk.StringVar(value="A")

        self.option_scores = {"A": {}, "B": {}}
        self.chosen_key = "A"
        self.other_key = "B"
        self.chosen_label = "Option A"
        self.other_label = "Option B"

        self.decision_scale = "standard"
        self.cognitive_questions = []
        self.option_questions = []
        self.current_phase = "cognitive"
        self.current_index = 0
        self.current_question = None
        self.asked_question_ids = set()
        self.total_steps_estimate = 1
        self.completed_steps = 0
        self.detected_biases = []

        self.intro()

    # ======================================================
    # SECTION 3: GENERIC UI HELPERS
    # ======================================================

    def clear(self):
        for widget in self.root.winfo_children():
            widget.destroy()

    # ======================================================
    # SECTION 4: INPUT + CONTEXT PREP
    # ======================================================

    def detect_context(self, decision_text):
        """Map free-text decision into one of the supported contexts."""
        decision_lower = decision_text.lower()
        for context_name, keywords in CONTEXT_KEYWORDS.items():
            if any(word in decision_lower for word in keywords):
                return context_name
        return "generic"

    def detect_decision_scale(self, decision_text, context):
        """Detect whether the decision is small, standard, or major using simple keyword rules."""
        decision_lower = decision_text.lower()

        if context in ["relationship", "career", "finance", "health"]:
            return "major"

        if any(word in decision_lower for word in MAJOR_DECISION_KEYWORDS):
            return "major"

        if any(word in decision_lower for word in SMALL_DECISION_KEYWORDS):
            return "small"

        return "standard"

    def _infer_options_from_decision(self, decision_text):
        """Infer options when user writes dilemma as 'X or Y'."""
        decision_clean = re.sub(r"\s+", " ", decision_text.strip())
        split_match = re.search(r"\b(.+?)\s+or\s+(.+)$", decision_clean, flags=re.IGNORECASE)
        if not split_match:
            return None, None

        left = split_match.group(1)
        right = split_match.group(2)

        left = re.sub(r"^(should i|do i|is it better to|would it be better to)\s+", "", left, flags=re.IGNORECASE).strip(" ?.,")
        right = right.strip(" ?.,")

        if not left or not right:
            return None, None

        return left.capitalize(), right.capitalize()

    def _identify_dilemma_profile(self, decision_text):
        """Identify most likely dilemma domain and generate an explainable profile."""
        text = decision_text.lower()
        scores = {}
        for domain, words in CONTEXT_KEYWORDS.items():
            scores[domain] = sum(1 for word in words if word in text)

        best_domain = max(scores, key=scores.get) if scores else "generic"
        best_score = scores.get(best_domain, 0)
        sorted_scores = sorted(scores.values(), reverse=True)
        second_best = sorted_scores[1] if len(sorted_scores) > 1 else 0

        if best_score == 0:
            best_domain = "generic"
            confidence = "low"
        elif best_score - second_best >= 2:
            confidence = "high"
        else:
            confidence = "medium"

        scale = self.detect_decision_scale(decision_text, best_domain)
        return {
            "domain": best_domain,
            "confidence": confidence,
            "scale": scale,
            "summary": f"{best_domain.title()} / {scale.title()}-impact",
        }

    def _counter_prompt(self):
        """Clear, concrete wording for opposite-case question."""
        chosen = self.option_a if self.leaning_var.get() == "A" else self.option_b
        other = self.option_b if self.leaning_var.get() == "A" else self.option_a
        prompt = f"How strong is the best case for '{other}' instead of '{chosen}'?"
        hint = (
            "0 = almost no case for the other option, 10 = very strong case for the other option"
        )
        return prompt, hint

    def _collect_intro_inputs(self):
        """Collect the first-screen input values and normalize defaults."""
        self.decision = self.decision_text.get("1.0", tk.END).strip() or "Undescribed decision"
        raw_option_a = self.option_a_var.get().strip()
        raw_option_b = self.option_b_var.get().strip()

        inferred_a, inferred_b = self._infer_options_from_decision(self.decision)
        self.option_a = raw_option_a or inferred_a or "Option A"
        self.option_b = raw_option_b or inferred_b or "Option B"

        self.dilemma_profile = self._identify_dilemma_profile(self.decision)
        self.context = self.dilemma_profile["domain"]
        self.decision_scale = self.dilemma_profile["scale"]

    # ======================================================
    # SECTION 5: SCREEN BUILDERS
    # ======================================================

    def intro(self):
        """First screen: collect decision statement and top-two options."""
        self.clear()
        page = tk.Frame(self.root, bg="#f3f5fb")
        page.pack(fill="both", expand=True, padx=24, pady=18)

        tk.Label(page, text="BiasLab", font=("Segoe UI", 30, "bold"), bg="#f3f5fb", fg="#0f172a").pack(anchor="w")
        tk.Label(
            page,
            text="Adaptive Decision Intelligence",
            font=("Segoe UI", 12, "bold"),
            bg="#f3f5fb",
            fg="#334155",
        ).pack(anchor="w", pady=(2, 0))
        tk.Label(
            page,
            text="Type your dilemma once. BiasLab will ask questions one-by-one and adapt based on your answers.",
            font=("Segoe UI", 10),
            bg="#f3f5fb",
            fg="#475569",
        ).pack(anchor="w", pady=(0, 12))

        card = tk.Frame(page, bg="white", highlightbackground="#dbe3f1", highlightthickness=1)
        card.pack(fill="both", expand=True)

        tk.Label(card, text="Decision Description", font=("Segoe UI", 12, "bold"), bg="white", fg="#0f172a").pack(anchor="w", padx=18, pady=(16, 4))
        tk.Label(
            card,
            text="Example: Should I approach my crush now or wait for better timing?",
            font=("Segoe UI", 9),
            bg="white",
            fg="#64748b",
        ).pack(anchor="w", padx=18, pady=(0, 4))
        self.decision_text = tk.Text(card, height=6, width=110, bg="#f8fafc", fg="#0f172a")
        self.decision_text.pack(padx=18, pady=(0, 10))

        option_frame = tk.Frame(card, bg="white")
        option_frame.pack(fill="x", padx=18, pady=(0, 8))
        tk.Label(option_frame, text="Option A", width=12, anchor="w", bg="white", fg="#1e293b", font=("Segoe UI", 10, "bold")).grid(row=0, column=0, padx=6, pady=5)
        tk.Entry(option_frame, textvariable=self.option_a_var, width=46, bg="#f8fafc").grid(row=0, column=1, padx=6, pady=5)
        tk.Label(option_frame, text="Option B", width=12, anchor="w", bg="white", fg="#1e293b", font=("Segoe UI", 10, "bold")).grid(row=1, column=0, padx=6, pady=5)
        tk.Entry(option_frame, textvariable=self.option_b_var, width=46, bg="#f8fafc").grid(row=1, column=1, padx=6, pady=5)

        leaning_frame = tk.Frame(card, bg="white")
        leaning_frame.pack(anchor="w", padx=18, pady=(6, 8))
        tk.Label(leaning_frame, text="Current leaning", bg="white", fg="#1e293b", font=("Segoe UI", 10, "bold")).pack(side="left", padx=(0, 10))
        ttk.Radiobutton(leaning_frame, text="Option A", variable=self.leaning_var, value="A").pack(side="left", padx=8)
        ttk.Radiobutton(leaning_frame, text="Option B", variable=self.leaning_var, value="B").pack(side="left", padx=8)

        tk.Label(
            card,
            text="Tip: If you write 'X or Y' in the dilemma text, BiasLab auto-detects both options.",
            font=("Segoe UI", 9),
            bg="white",
            fg="#64748b",
        ).pack(anchor="w", padx=18, pady=(0, 12))

        tk.Button(
            card,
            text="Start Adaptive Analysis",
            command=self.open_deep_assessment,
            font=("Segoe UI", 10, "bold"),
            bg="#2563eb",
            fg="white",
            relief="flat",
            padx=16,
            pady=8,
        ).pack(anchor="e", padx=18, pady=(0, 16))

    def _criterion_prompt(self, key):
        prompt_map = {
            "need_fit": "How helpful is this option for what you actually want right now?",
            "long_term": "How good is this option after 6-12 months?",
            "tradeoff": "Is this option worth what you must give up (money/time/effort)?",
            "evidence": "How much real proof supports this option (facts, patterns, data)?",
            "compatibility": "How easily does this option fit your current life/system?",
        }
        return prompt_map.get(key, key)

    def _customize_question_wording(self, questions):
        """Make prompts concrete for detected dilemma domain and selected options."""
        prompt_overrides = {}
        hint_overrides = {}

        opposite_prompt, opposite_hint = self._counter_prompt()
        prompt_overrides["counter_strength"] = opposite_prompt
        hint_overrides["counter_strength"] = opposite_hint

        if self.context == "relationship":
            prompt_overrides.update(
                {
                    "emotion": "How emotionally affected are you when you think about this person/situation?",
                    "social_pressure": "How much are friends/family/social opinions shaping this choice?",
                    "fairness": "How respectful is your current choice for both people involved?",
                    "alt_exploration": "Have you seriously considered both paths (approach vs not approach / continue vs stop)?",
                }
            )

        if self.context == "career":
            prompt_overrides.update(
                {
                    "social_pressure": "How much are status/parents/society pushing this choice?",
                    "identity_attachment": "How much is this tied to your image of success?",
                    "evidence": "How strong are real facts (market, mentors, outcomes) supporting this?",
                }
            )

        if self.context == "purchase":
            prompt_overrides.update(
                {
                    "tradeoff": "How good is value-for-money in this option?",
                    "compatibility": "How well does this fit your existing setup/ecosystem?",
                }
            )

        if self.context == "health":
            prompt_overrides.update(
                {
                    "emotion": "How much fear/anxiety is driving this health choice?",
                    "evidence": "How strongly is this backed by trusted health advice/data?",
                    "harm_risk": "If this choice is wrong, how much could it hurt your health?",
                }
            )

        if self.context == "academic":
            prompt_overrides.update(
                {
                    "urgency": "How rushed do you feel because of deadlines/exams?",
                    "alt_exploration": "How much did you compare alternatives (course/plan/path) properly?",
                }
            )

        for question in questions:
            key = question.get("key")
            if key in prompt_overrides:
                question["prompt"] = prompt_overrides[key]
            if key in hint_overrides:
                question["hint"] = hint_overrides[key]

        return questions

    def _base_cognitive_questions(self):
        """Base question set selected by context + decision scale (common categories only)."""
        if self.decision_scale == "small":
            questions = [
                {
                    "id": "emotion",
                    "type": "single_scale",
                    "key": "emotion",
                    "prompt": "How emotionally loaded are you about this decision right now?",
                    "hint": "0 = calm, 10 = very emotional",
                    "default": 3,
                },
                {
                    "id": "urgency",
                    "type": "single_scale",
                    "key": "urgency",
                    "prompt": "How rushed do you feel to decide now?",
                    "hint": "0 = no rush, 10 = extreme rush",
                    "default": 3,
                },
                {
                    "id": "social_pressure",
                    "type": "single_scale",
                    "key": "social_pressure",
                    "prompt": "How much are others pushing you toward one option?",
                    "hint": "0 = no pressure from others, 10 = very strong pressure",
                    "default": 2,
                },
                {
                    "id": "counter_strength",
                    "type": "single_scale",
                    "key": "counter_strength",
                    "prompt": "How strong is the best case for the other option?",
                    "hint": "0 = almost no case, 10 = very strong case",
                    "default": 5,
                },
                {
                    "id": "alt_exploration",
                    "type": "single_scale",
                    "key": "alt_exploration",
                    "prompt": "Did you genuinely check at least one other option?",
                    "hint": "0 = not at all, 10 = yes carefully",
                    "default": 5,
                },
                {
                    "id": "fairness",
                    "type": "single_scale",
                    "key": "fairness",
                    "prompt": "How respectful and fair is this choice to everyone involved?",
                    "hint": "0 = unfair/disrespectful, 10 = very fair/respectful",
                    "default": 8,
                },
            ]
            if self.context == "purchase":
                questions.insert(
                    3,
                    {
                        "id": "novelty_pull",
                        "type": "single_scale",
                        "key": "novelty_pull",
                        "prompt": "Are you choosing mostly because it feels new/exciting?",
                        "hint": "0 = no, 10 = mostly yes",
                        "default": 4,
                    },
                )
            return self._customize_question_wording(questions)

        base_questions = [
            {
                "id": "emotion",
                "type": "single_scale",
                "key": "emotion",
                "prompt": "How emotionally charged do you feel about this decision?",
                "hint": "0 = calm, 10 = highly emotional",
                "default": 5,
            },
            {
                "id": "urgency",
                "type": "single_scale",
                "key": "urgency",
                "prompt": "How much pressure do you feel to decide fast?",
                "hint": "0 = no pressure, 10 = very rushed",
                "default": 5,
            },
            {
                "id": "social_pressure",
                "type": "single_scale",
                "key": "social_pressure",
                "prompt": "How much are family/friends/society pushing your choice?",
                "hint": "0 = no influence, 10 = very strong influence",
                "default": 4,
            },
            {
                "id": "counter_strength",
                "type": "single_scale",
                "key": "counter_strength",
                "prompt": "How strong is the best case for the opposite option?",
                "hint": "0 = very weak, 10 = very strong",
                "default": 5,
            },
            {
                "id": "alt_exploration",
                "type": "single_scale",
                "key": "alt_exploration",
                "prompt": "How seriously did you compare other options?",
                "hint": "0 = barely compared, 10 = compared thoroughly",
                "default": 5,
            },
            {
                "id": "fairness",
                "type": "single_scale",
                "key": "fairness",
                "prompt": "How respectful and fair is your current choice to all affected people?",
                "hint": "0 = unfair/disrespectful, 10 = very fair/respectful",
                "default": 7,
            },
            {
                "id": "harm_risk",
                "type": "single_scale",
                "key": "harm_risk",
                "prompt": "If this choice turns out wrong, how serious could the damage be?",
                "hint": "0 = almost no harm, 10 = severe harm",
                "default": 5,
            },
        ]

        if self.decision_scale == "major":
            major_additions = [
                {
                    "id": "identity_attachment",
                    "type": "single_scale",
                    "key": "identity_attachment",
                    "prompt": "How much is your self-image attached to one option?",
                    "hint": "0 = not attached, 10 = strongly attached",
                    "default": 5,
                },
                {
                    "id": "sunk_cost",
                    "type": "single_scale",
                    "key": "sunk_cost",
                    "prompt": "How much are past time/money efforts trapping you in this choice?",
                    "hint": "0 = no effect, 10 = very strong effect",
                    "default": 5,
                },
                {
                    "id": "loss_aversion",
                    "type": "single_scale",
                    "key": "loss_aversion",
                    "prompt": "How much fear of losing something is driving this choice?",
                    "hint": "0 = not at all, 10 = very much",
                    "default": 5,
                },
                {
                    "id": "failure_preview",
                    "type": "single_scale",
                    "key": "failure_preview",
                    "prompt": "How clearly can you imagine this choice going badly?",
                    "hint": "0 = cannot imagine, 10 = very clearly",
                    "default": 5,
                },
                {
                    "id": "regret_preview",
                    "type": "single_scale",
                    "key": "regret_preview",
                    "prompt": "How clearly can you imagine regretting this later?",
                    "hint": "0 = unclear, 10 = very clear",
                    "default": 5,
                },
            ]
            base_questions[3:3] = major_additions

        return self._customize_question_wording(base_questions)

    def _build_option_questions(self):
        """Adaptive option-comparison questions. One criterion per step, two sliders inside."""
        if self.decision_scale == "small":
            keys = ["need_fit", "tradeoff", "evidence"]
            if self.context == "purchase":
                keys.append("compatibility")
        elif self.decision_scale == "major":
            keys = ["need_fit", "evidence", "long_term", "compatibility", "tradeoff"]
        else:
            keys = ["need_fit", "evidence", "tradeoff", "compatibility"]

        if self.context == "relationship" and "long_term" not in keys:
            keys.insert(2, "long_term")

        questions = []
        for key in keys:
            questions.append(
                {
                    "id": f"pair_{key}",
                    "type": "pair_scale",
                    "key": key,
                    "prompt": self._criterion_prompt(key),
                    "hint": f"Rate {self.option_a} and {self.option_b} separately on this criterion.",
                    "default": 5,
                }
            )

        questions.append(
            {
                "id": "reason_a",
                "type": "text",
                "key": "reason_a",
                "prompt": f"In one sentence: best practical reason to choose {self.option_a}",
                "hint": "Use practical reasons, not vibes.",
                "required": False,
            }
        )
        questions.append(
            {
                "id": "reason_b",
                "type": "text",
                "key": "reason_b",
                "prompt": f"In one sentence: best practical reason to choose {self.option_b}",
                "hint": "Use practical reasons, not vibes.",
                "required": False,
            }
        )
        return questions

    def _append_question_if_new(self, question):
        existing_ids = {q["id"] for q in self.cognitive_questions}
        if question["id"] not in existing_ids and question["id"] not in self.asked_question_ids:
            self.cognitive_questions.append(question)
            self.total_steps_estimate += 1

    def _inject_followups(self, question, normalized_value):
        """Add targeted follow-up prompts based on previous answers."""
        key = question.get("key")

        if key == "emotion" and normalized_value >= 0.70:
            self._append_question_if_new(
                {
                    "id": "emotion_source_note",
                    "type": "text",
                    "key": "emotion_source_note",
                    "prompt": "What exactly is creating this strong emotion?",
                    "hint": "Naming it helps reduce hidden bias.",
                    "required": True,
                }
            )

        if key == "social_pressure" and normalized_value >= 0.60:
            self._append_question_if_new(
                {
                    "id": "social_source_note",
                    "type": "text",
                    "key": "social_source_note",
                    "prompt": "Who is influencing your decision the most right now?",
                    "hint": "Be specific.",
                    "required": True,
                }
            )

        if key == "urgency" and normalized_value >= 0.65:
            self._append_question_if_new(
                {
                    "id": "urgency_reason_note",
                    "type": "text",
                    "key": "urgency_reason_note",
                    "prompt": "Is this urgency truly real, or are you creating it yourself?",
                    "hint": "Write one sentence.",
                    "required": True,
                }
            )

        if key == "counter_strength" and normalized_value <= 0.45:
            self._append_question_if_new(
                {
                    "id": "counter_text",
                    "type": "text",
                    "key": "counter_text",
                    "prompt": "Write one strong reason your current preferred option might be wrong.",
                    "hint": "This is required for bias resistance.",
                    "required": True,
                }
            )

        if key == "fairness" and normalized_value <= 0.45:
            self._append_question_if_new(
                {
                    "id": "harm_risk",
                    "type": "single_scale",
                    "key": "harm_risk",
                    "prompt": "If your choice is unfair, how much harm could it cause others?",
                    "hint": "0 = almost none, 10 = severe harm",
                    "default": 6,
                }
            )

    def _get_active_questions(self):
        return self.cognitive_questions if self.current_phase == "cognitive" else self.option_questions

    def _header_copy(self):
        if self.current_phase == "cognitive":
            return "Adaptive Bias Questions"
        return "Adaptive Option Comparison"

    def _phase_copy(self):
        if self.current_phase == "cognitive":
            profile = getattr(self, "dilemma_profile", None)
            if profile:
                return (
                    f"Detected dilemma: {profile['summary']} | "
                    f"Confidence: {profile['confidence'].title()}"
                )
            return f"Detected: {self.context.title()} decision ({self.decision_scale})"
        return f"Compare options carefully: {self.option_a} vs {self.option_b}"

    def _render_question_screen(self):
        """Render one question at a time with a cleaner card-style UI."""
        self.clear()

        active_questions = self._get_active_questions()
        if self.current_index >= len(active_questions):
            if self.current_phase == "cognitive":
                self.current_phase = "options"
                self.current_index = 0
                self._render_question_screen()
                return
            self.compute_analysis()
            return

        self.current_question = active_questions[self.current_index]

        root_wrap = tk.Frame(self.root, bg="#f3f5fb")
        root_wrap.pack(fill="both", expand=True, padx=24, pady=16)

        tk.Label(root_wrap, text="BiasLab", font=("Segoe UI", 24, "bold"), bg="#f3f5fb", fg="#1f2937").pack(anchor="w")
        tk.Label(root_wrap, text=self._header_copy(), font=("Segoe UI", 12, "bold"), bg="#f3f5fb", fg="#334155").pack(anchor="w", pady=(3, 0))
        tk.Label(root_wrap, text=self._phase_copy(), font=("Segoe UI", 10), bg="#f3f5fb", fg="#475569").pack(anchor="w", pady=(0, 8))

        progress_pct = int((self.completed_steps / max(self.total_steps_estimate, 1)) * 100)
        ttk.Progressbar(root_wrap, mode="determinate", value=progress_pct, maximum=100, length=760).pack(anchor="w", pady=(0, 10))

        card = tk.Frame(root_wrap, bg="white", highlightbackground="#dbe3f1", highlightthickness=1)
        card.pack(fill="both", expand=True)

        q = self.current_question
        tk.Label(card, text=f"Question {self.completed_steps + 1}", font=("Segoe UI", 10, "bold"), bg="white", fg="#2563eb").pack(anchor="w", padx=18, pady=(16, 2))
        tk.Label(card, text=q["prompt"], font=("Segoe UI", 14, "bold"), bg="white", fg="#111827", wraplength=980, justify="left").pack(anchor="w", padx=18, pady=(0, 4))
        tk.Label(card, text=q.get("hint", ""), font=("Segoe UI", 10), bg="white", fg="#6b7280", wraplength=980, justify="left").pack(anchor="w", padx=18, pady=(0, 12))

        self.current_error_label = tk.Label(card, text="", font=("Segoe UI", 10), bg="white", fg="#b91c1c")
        self.current_error_label.pack(anchor="w", padx=18)

        self.current_scale_widget = None
        self.current_scale_a_widget = None
        self.current_scale_b_widget = None
        self.current_text_widget = None

        if q["type"] == "single_scale":
            self.current_scale_widget = tk.Scale(card, from_=0, to=10, orient="horizontal", length=560, bg="white")
            self.current_scale_widget.set(q.get("default", 5))
            self.current_scale_widget.pack(anchor="w", padx=18, pady=(8, 18))

        elif q["type"] == "pair_scale":
            row_a = tk.Frame(card, bg="white")
            row_a.pack(fill="x", padx=18, pady=(6, 6))
            tk.Label(row_a, text=self.option_a, width=18, anchor="w", bg="white", fg="#111827", font=("Segoe UI", 10, "bold")).pack(side="left")
            self.current_scale_a_widget = tk.Scale(row_a, from_=0, to=10, orient="horizontal", length=460, bg="white")
            self.current_scale_a_widget.set(q.get("default", 5))
            self.current_scale_a_widget.pack(side="left")

            row_b = tk.Frame(card, bg="white")
            row_b.pack(fill="x", padx=18, pady=(0, 14))
            tk.Label(row_b, text=self.option_b, width=18, anchor="w", bg="white", fg="#111827", font=("Segoe UI", 10, "bold")).pack(side="left")
            self.current_scale_b_widget = tk.Scale(row_b, from_=0, to=10, orient="horizontal", length=460, bg="white")
            self.current_scale_b_widget.set(q.get("default", 5))
            self.current_scale_b_widget.pack(side="left")

        elif q["type"] == "text":
            self.current_text_widget = tk.Text(card, height=5, width=110, bg="#f8fafc", fg="#0f172a")
            self.current_text_widget.pack(anchor="w", padx=18, pady=(8, 18))

        footer = tk.Frame(card, bg="white")
        footer.pack(fill="x", padx=18, pady=(4, 16))
        tk.Label(footer, text=f"Progress: {self.completed_steps}/{self.total_steps_estimate}", bg="white", fg="#64748b").pack(side="left")

        button_text = "Next" if (self.current_phase == "cognitive" or self.current_index < len(active_questions) - 1) else "Analyze Decision"
        tk.Button(
            footer,
            text=button_text,
            command=self._submit_current_question,
            font=("Segoe UI", 10, "bold"),
            bg="#2563eb",
            fg="white",
            padx=16,
            pady=6,
            relief="flat",
        ).pack(side="right")

    def _submit_current_question(self):
        q = self.current_question
        if q["type"] == "single_scale":
            normalized = normalize(self.current_scale_widget.get())
            self.answers[q["key"]] = normalized
            self._inject_followups(q, normalized)

        elif q["type"] == "pair_scale":
            self.option_scores["A"][q["key"]] = normalize(self.current_scale_a_widget.get())
            self.option_scores["B"][q["key"]] = normalize(self.current_scale_b_widget.get())

        elif q["type"] == "text":
            text_value = self.current_text_widget.get("1.0", tk.END).strip()
            if q.get("required") and not text_value:
                self.current_error_label.config(text="This answer is required to continue.")
                return
            self.answers[q["key"]] = text_value

        self.asked_question_ids.add(q["id"])
        self.completed_steps += 1
        self.current_index += 1
        self._render_question_screen()

    def open_deep_assessment(self):
        """Adaptive pipeline: one-question-at-a-time, with dynamic follow-ups."""
        self._collect_intro_inputs()

        self.answers = {}
        self.option_scores = {"A": {}, "B": {}}
        self.cognitive_questions = self._base_cognitive_questions()
        self.option_questions = self._build_option_questions()
        self.current_phase = "cognitive"
        self.current_index = 0
        self.completed_steps = 0
        self.asked_question_ids = set()
        self.total_steps_estimate = len(self.cognitive_questions) + len(self.option_questions)

        self._render_question_screen()

    # ======================================================
    # SECTION 6: DATA COLLECTION HELPERS
    # ======================================================

    def _average(self, values):
        return sum(values) / len(values) if values else 0.0

    def _answer(self, key, default=0.5):
        """Safe accessor for slider answers when a personalized question set omits some keys."""
        return self.answers.get(key, default)

    # ======================================================
    # SECTION 7: MATH + SCORING ENGINE (ALL TECHNICALS)
    # ======================================================

    def _calculate_rational_quality(self, scores):
        """Weighted rational score for one option."""
        return sum(scores[key] * weight for key, weight in RATIONAL_WEIGHTS.items())

    def _calculate_bias_pressure(self):
        """Average pressure from affective/cognitive distortion signals."""
        values = [self._answer(key) for key in BIAS_PRESSURE_KEYS]
        values.append(1 - self._answer("counter_strength"))
        return self._average(values)

    def _calculate_foresight_gap(self):
        """How weakly future consequences were explored."""
        return self._average(
            [
                1 - self._answer("failure_preview"),
                1 - self._answer("regret_preview"),
                1 - self._answer("alt_exploration"),
            ]
        )

    def _calculate_fairness_risk(self):
        """Blend fairness deficit and harm risk."""
        return clamp01((1 - self._answer("fairness")) * 0.60 + self._answer("harm_risk") * 0.40)

    def _calculate_distortion_risk(
        self,
        bias_pressure,
        foresight_gap,
        fairness_risk,
        weak_choice_penalty,
        chosen_rational,
    ):
        """Final distortion risk composed from all core penalties."""
        return clamp01(
            bias_pressure * DISTORTION_WEIGHTS["bias_pressure"]
            + foresight_gap * DISTORTION_WEIGHTS["foresight_gap"]
            + fairness_risk * DISTORTION_WEIGHTS["fairness_risk"]
            + weak_choice_penalty * DISTORTION_WEIGHTS["weak_choice_penalty"]
            + (1 - chosen_rational) * DISTORTION_WEIGHTS["low_evidence_penalty"]
        )

    def _is_practically_justified(self, chosen_scores, justification_gap):
        """Safeguard that prevents falsely labeling practical preferences as bias."""
        return (
            chosen_scores.get("compatibility", 0.5) >= 0.70
            and chosen_scores.get("need_fit", 0.5) >= 0.65
            and chosen_scores.get("evidence", 0.5) >= 0.55
            and justification_gap >= 0.05
            and self._answer("counter_strength") >= 0.45
        )

    def _assign_choice_labels(self, chosen_key):
        self.chosen_key = chosen_key
        self.other_key = "B" if chosen_key == "A" else "A"
        self.chosen_label = self.option_a if chosen_key == "A" else self.option_b
        self.other_label = self.option_b if chosen_key == "A" else self.option_a

    def _detect_bias_patterns(self):
        """Detect likely cognitive-bias patterns from scored signals and answers."""
        rules = [
            {
                "name": "Emotional Reasoning",
                "score": lambda: self._answer("emotion"),
                "threshold": 0.70,
                "reality": "Your feelings are so strong they may be driving the choice.",
                "action": "Wait for emotions to cool down, then re-answer the questions.",
            },
            {
                "name": "Social Pressure Bias",
                "score": lambda: self._answer("social_pressure"),
                "threshold": 0.65,
                "reality": "Other people may be pushing your choice more than your own values.",
                "action": "Decide in private first, then compare with outside opinions.",
            },
            {
                "name": "Sunk Cost Fallacy",
                "score": lambda: self._answer("sunk_cost"),
                "threshold": 0.60,
                "reality": "Past time/money may be trapping you in this choice.",
                "action": "Ask: 'If I started today, would I still choose this?'",
            },
            {
                "name": "Identity Attachment Bias",
                "score": lambda: self._answer("identity_attachment"),
                "threshold": 0.65,
                "reality": "Your self-image may be tied to one option.",
                "action": "Imagine you are advising a close friend with the same facts.",
            },
            {
                "name": "Loss Aversion Bias",
                "score": lambda: self._answer("loss_aversion"),
                "threshold": 0.65,
                "reality": "Fear of loss may be louder than real upside/downside balance.",
                "action": "List likely losses and likely gains side by side.",
            },
            {
                "name": "Novelty Attraction Bias",
                "score": lambda: self._answer("novelty_pull"),
                "threshold": 0.65,
                "reality": "Newness/excitement may be making one option look better than it is.",
                "action": "Re-score options while ignoring excitement and focusing on outcomes.",
            },
            {
                "name": "Confirmation / Tunnel Vision",
                "score": lambda: max(1 - self._answer("counter_strength"), 1 - self._answer("alt_exploration")),
                "threshold": 0.55,
                "reality": "You may be focusing too much on one side and not testing the other.",
                "action": "Write the strongest argument for the opposite option.",
            },
            {
                "name": "Outcome Blindness (Optimism Bias)",
                "score": lambda: ((1 - self._answer("failure_preview")) + (1 - self._answer("regret_preview"))) / 2,
                "threshold": 0.60,
                "reality": "You may be underthinking how this could go wrong.",
                "action": "Write a worst-case story and how you would handle it.",
            },
            {
                "name": "Fairness Blind Spot",
                "score": lambda: max(1 - self._answer("fairness"), self._answer("harm_risk")),
                "threshold": 0.60,
                "reality": "You may be underweighting how this affects other people.",
                "action": "List who is affected and how your choice changes their life.",
            },
            {
                "name": "Weak-Evidence Decision Bias",
                "score": lambda: max(1 - self.chosen_rational, self.signal_map.get("Low Evidence Penalty", 0)),
                "threshold": 0.60,
                "reality": "Your choice may not be backed by enough real proof yet.",
                "action": "Collect 2-3 concrete facts before fully committing.",
            },
        ]

        hits = []
        for rule in rules:
            score = rule["score"]()
            if score >= rule["threshold"]:
                hits.append(
                    {
                        "name": rule["name"],
                        "score": score,
                        "reality": rule["reality"],
                        "action": rule["action"],
                    }
                )

        hits.sort(key=lambda item: item["score"], reverse=True)
        return hits[:4]

    def _reality_check_lines(self):
        lines = ["2) Reality Check"]

        if self.justification_gap >= 0.10 and self.total_risk <= 0.45:
            lines.append(
                "- Reality: your current option is backed more by real reasons than by pressure."
            )
        elif self.justification_gap < 0 and self.total_risk >= 0.50:
            lines.append(
                "- Reality: your current leaning looks more driven by pressure than by real reasons."
            )
        else:
            lines.append(
                "- Reality: the decision is mixed; some reasons are solid, some are shaky."
            )

        lines.append(
            f"- Snapshot: gap between options = {self.justification_gap:.2f}, bias risk = {self.total_risk:.2f}."
        )
        lines.append("")
        return lines

    def _bias_pattern_lines(self):
        lines = ["4) Likely Biases"]
        if not self.detected_biases:
            lines.append("- No strong bias pattern detected. Still verify real evidence.")
            lines.append("")
            return lines

        for bias in self.detected_biases:
            lines.append(f"- {bias['name']} ({bias['score']:.2f}): {bias['reality']}")
        lines.append("")
        return lines

    def _bias_solution_lines(self):
        lines = ["5) What You Should Do Next"]

        if self.detected_biases:
            for bias in self.detected_biases[:3]:
                lines.append(f"- For {bias['name']}: {bias['action']}")
        else:
            lines.append("- Keep your plan, but confirm at least one more real fact first.")

        lines.append("- Re-run BiasLab after applying the steps above and compare distortion risk change.")
        lines.append("")
        return lines

    def compute_analysis(self):
        """Primary compute pipeline: collect data -> run scoring -> build report state."""
        chosen = self.leaning_var.get()
        self._assign_choice_labels(chosen)

        chosen_scores = self.option_scores.get(self.chosen_key, {})
        other_scores = self.option_scores.get(self.other_key, {})

        for key in OPTION_CRITERIA_KEYS:
            chosen_scores.setdefault(key, 0.5)
            other_scores.setdefault(key, 0.5)

        chosen_rational = self._calculate_rational_quality(chosen_scores)
        other_rational = self._calculate_rational_quality(other_scores)
        bias_pressure = self._calculate_bias_pressure()
        foresight_gap = self._calculate_foresight_gap()
        fairness_risk = self._calculate_fairness_risk()

        justification_gap = chosen_rational - other_rational
        weak_choice_penalty = max(0.0, -justification_gap)
        distortion_risk = self._calculate_distortion_risk(
            bias_pressure,
            foresight_gap,
            fairness_risk,
            weak_choice_penalty,
            chosen_rational,
        )

        practical_preference = self._is_practically_justified(chosen_scores, justification_gap)

        self.total_risk = distortion_risk
        self.integrity = 1 - distortion_risk
        self.chosen_rational = chosen_rational
        self.other_rational = other_rational
        self.justification_gap = justification_gap
        self.practical_preference = practical_preference

        self.signal_map = {
            "Bias Pressure": bias_pressure,
            "Foresight Gap": foresight_gap,
            "Fairness Risk": fairness_risk,
            "Weak Choice Penalty": weak_choice_penalty,
            "Low Evidence Penalty": (1 - chosen_rational),
        }

        self.detected_biases = self._detect_bias_patterns()

        self.report()

    # ======================================================
    # SECTION 8: REPORT TEXT ENGINE (ALL OUTPUT SENTENCES)
    # ======================================================

    def _summary_lines(self):
        return [
            "1) Quick Summary",
            f"- Overall signal: {classify_risk(self.total_risk)}",
            f"- Bias risk: {self.total_risk:.2f} | Clarity score: {self.integrity:.2f}",
            f"- Strength of {self.chosen_label}: {self.chosen_rational:.2f}",
            f"- Strength of {self.other_label}: {self.other_rational:.2f}",
            f"- Gap between options: {self.justification_gap:.2f}",
            "",
        ]

    def _driver_lines(self):
        lines = ["3) What is pushing your choice"]
        items = sorted(self.signal_map.items(), key=lambda item: item[1], reverse=True)[:4]
        for name, value in items:
            level = "high" if value >= 0.70 else "moderate" if value >= 0.45 else "low"
            lines.append(f"- {name}: {value:.2f} ({level} impact)")
        lines.append("")
        return lines

    def _interpretation_lines(self):
        lines = ["6) Plain-English Verdict"]
        if self.practical_preference:
            lines.append(
                f"- Your choice of {self.chosen_label} looks reasonable, not just emotional."
            )
            lines.append("- It matches your needs, evidence, and fit with your life.")
        elif self.justification_gap >= 0:
            lines.append("- Your choice has some good reasons, but there is also strong pressure.")
        else:
            lines.append("- The other option looks stronger on real reasons; this is a warning sign.")
        lines.append("")
        return lines

    def _action_protocol_lines(self):
        lines = ["7) Simple Next Steps"]
        if self.signal_map["Bias Pressure"] > 0.60:
            lines.append("- Wait 24 hours, then answer the same questions again with a calm mind.")
        if self.signal_map["Foresight Gap"] > 0.50:
            lines.append("- Write a best-case and worst-case story for both options.")
        if self.signal_map["Fairness Risk"] > 0.50:
            lines.append("- Check if your choice is respectful to everyone involved.")
        if self.signal_map["Weak Choice Penalty"] > 0.40:
            lines.append("- Seriously test the other option before committing.")
        lines.append("- Re-run BiasLab after you learn 1-2 new facts.")
        lines.append("")
        return lines

    def _reflection_lines(self):
        lines = ["8) Your Notes"]
        if self.answers.get("counter_text"):
            lines.append(f"- Strongest counter-argument captured: {self.answers['counter_text']}")
        if self.answers.get("reason_a"):
            lines.append(f"- Practical reason for {self.option_a}: {self.answers['reason_a']}")
        if self.answers.get("reason_b"):
            lines.append(f"- Practical reason for {self.option_b}: {self.answers['reason_b']}")
        return lines

    def generate_narrative(self):
        """Assemble final report text from section-specific text builders."""
        lines = [
            "BiasLab Practical Decision Report",
            "=" * 40,
            f"Decision: {self.decision}",
            f"Current leaning: {self.chosen_label}",
            f"Alternative: {self.other_label}",
            "",
        ]
        lines.extend(self._summary_lines())
        lines.extend(self._reality_check_lines())
        lines.extend(self._driver_lines())
        lines.extend(self._bias_pattern_lines())
        lines.extend(self._bias_solution_lines())
        lines.extend(self._interpretation_lines())
        lines.extend(self._action_protocol_lines())
        lines.extend(self._reflection_lines())
        return "\n".join(lines)

    # ======================================================
    # SECTION 9: OUTPUT VIEWS (REPORT + CHART)
    # ======================================================

    def report(self):
        self.clear()
        tk.Label(self.root, text="Decision Analysis Report", font=("Helvetica", 20, "bold")).pack(pady=10)
        tk.Label(
            self.root,
            text=f"Status: {classify_risk(self.total_risk)} | Distortion Risk: {self.total_risk:.2f}",
            font=("Helvetica", 12),
        ).pack(pady=4)

        report_text = self.generate_narrative()

        text_box = tk.Text(self.root, height=27, width=125)
        text_box.pack(pady=8)
        text_box.insert(tk.END, report_text)
        text_box.config(state="disabled")

        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=8)
        tk.Button(button_frame, text="Show Bias Radar", command=self.show_radar).pack(side="left", padx=8)
        tk.Button(button_frame, text="Start New Decision", command=self.intro).pack(side="left", padx=8)

        self.save_session()

    def show_radar(self):
        """Visual chart of highest-level distortion components."""
        label_map = {
            "Bias Pressure": "Pressure / Emotions",
            "Foresight Gap": "Future Not Considered",
            "Fairness Risk": "Fairness Risk",
            "Weak Choice Penalty": "Other Option Looks Stronger",
            "Low Evidence Penalty": "Low Real Evidence",
        }

        labels = [label_map.get(key, key) for key in self.signal_map.keys()]
        values = list(self.signal_map.values())

        values_cycle = values + values[:1]
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        angles_cycle = angles + angles[:1]

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True})
        ax.plot(angles_cycle, values_cycle, linewidth=2)
        ax.fill(angles_cycle, values_cycle, alpha=0.25)
        ax.set_xticks(angles)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1)
        plt.title("Decision Pressure Radar")
        plt.show()

    # ======================================================
    # SECTION 10: SESSION PERSISTENCE
    # ======================================================

    def save_session(self):
        file_name = "biaslab_sessions.csv"
        now = datetime.datetime.now().isoformat(timespec="seconds")

        needs_header = not os.path.exists(file_name)
        with open(file_name, "a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            if needs_header:
                writer.writerow(
                    [
                        "timestamp",
                        "decision",
                        "chosen_option",
                        "other_option",
                        "distortion_risk",
                        "integrity_score",
                        "chosen_rational",
                        "other_rational",
                        "justification_gap",
                        "practical_preference",
                    ]
                )

            writer.writerow(
                [
                    now,
                    self.decision,
                    self.chosen_label,
                    self.other_label,
                    round(self.total_risk, 4),
                    round(self.integrity, 4),
                    round(self.chosen_rational, 4),
                    round(self.other_rational, 4),
                    round(self.justification_gap, 4),
                    self.practical_preference,
                ]
            )


# ==========================================================
# SECTION 11: ENTRY POINT
# ==========================================================

root = tk.Tk()
app = BiasLab(root)
root.mainloop()
