"""
preprocessing.py — Stage A: Document prep, sentence tokenization, and context detection.
Splits text into `AnnotatedSentence` objects containing pre-computed context flags
(quotes, negation, attribution) so downstream scorers only process them once.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import List, Tuple
from biaslab.config import ATTRIBUTION_VERBS, NEGATION_WORDS, NEGATION_WINDOW

@dataclass
class AnnotatedSentence:
    """A single sentence with precomputed context flags."""
    text: str
    index: int = 0
    is_headline: bool = False
    
    # Context flags
    text_lower: str = field(init=False)
    words: List[str] = field(init=False)
    word_count: int = field(init=False)
    
    is_quoted: bool = field(init=False)
    has_quote: bool = field(init=False)
    quoted_spans: List[Tuple[int, int]] = field(init=False)
    
    is_attributed: bool = field(init=False)
    negation_positions: List[int] = field(init=False)

    def __post_init__(self):
        # 1. Normalize
        self.text_lower = re.sub(r"\s+", " ", self.text.lower()).strip()
        self.words = re.findall(r"\b[a-z']+\b", self.text_lower)
        self.word_count = max(len(self.words), 1)
        
        # 2. Quotes
        self.quoted_spans = _find_quoted_spans(self.text)
        self.has_quote = len(self.quoted_spans) > 0
        q_frac = sum(e - s for s, e in self.quoted_spans) / max(len(self.text), 1)
        self.is_quoted = q_frac > 0.50
        
        # 3. Attributes & Negations
        attr_regex = r"\b(" + "|".join(re.escape(v) for v in ATTRIBUTION_VERBS) + r")\b"
        self.is_attributed = bool(re.search(attr_regex, self.text_lower)) or "according to" in self.text_lower
        
        neg_set = set(NEGATION_WORDS)
        self.negation_positions = [
            i for i, w in enumerate(self.words)
            if w in neg_set or w.endswith("n't")
        ]

def _find_quoted_spans(text: str) -> List[Tuple[int, int]]:
    """Returns merged (start, end) indices of quoted substrings."""
    patterns = [r'"([^"\n]{4,}?)"', r'\u201C([^\u201D\n]{4,}?)\u201D', r"'([^'\n]{10,}?)'", r'\u2018([^\u2019\n]{10,}?)\u2019']
    spans = [m.span() for pat in patterns for m in re.finditer(pat, text)]
    if not spans: return []
    spans.sort()
    merged = [spans[0]]
    for s, e in spans[1:]:
        if s <= merged[-1][1]: merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else: merged.append((s, e))
    return merged

def _split_sentences(text: str) -> List[str]:
    """Robust sentence splitting using NLTK (or regex fallback)."""
    text = text.strip()
    try:
        import nltk  # type: ignore
        try: return nltk.sent_tokenize(text)
        except LookupError:
            nltk.download("punkt_tab", quiet=True)
            return nltk.sent_tokenize(text)
    except ImportError:
        # Regex fallback: split on punctuation followed by whitespace/quote and uppercase/quote.
        parts = re.split(r"(?<=[.!?])(?:\s+|(?=[\"\u201C]))(?=[A-Z\"\u201C])", text)
        return [p.strip() for p in parts if p.strip()]

def preprocess_article(title: str, body: str) -> Tuple[List[AnnotatedSentence], dict]:
    """Runs full Stage A pipeline. Returns (sentences, article_wide_stats)."""
    raw_title, raw_body = title.strip(), body.strip()
    sentences = []
    
    # Process Headline
    for s in _split_sentences(raw_title) if raw_title else []:
        if s.strip(): sentences.append(AnnotatedSentence(text=s, index=len(sentences), is_headline=True))
    
    # Process Body
    for s in _split_sentences(raw_body):
        if s.strip(): sentences.append(AnnotatedSentence(text=s, index=len(sentences), is_headline=False))

    all_text = f"{raw_title} {raw_body}"
    stats = {
        "n_words": sum(s.word_count for s in sentences) or 1,
        "n_sentences": len(sentences),
        "paragraphs": len([p for p in re.split(r"\n\s*\n", raw_body) if p.strip()]),
        "exclamations": all_text.count("!"),
        "questions": all_text.count("?"),
        "all_caps_words": len(re.findall(r"\b[A-Z]{4,}\b", all_text)),
        "quoted_segments": sum(1 for pat in [r'"[^"\n]{4,}?"', r'\u201C[^\u201D\n]{4,}?\u201D'] for _ in re.finditer(pat, raw_body)),
        "quoted_sentence_fraction": sum(1 for s in sentences if s.is_quoted or s.has_quote) / max(len(sentences), 1),
    }
    return sentences, stats
