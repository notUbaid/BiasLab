"""
confidence.py — Multi-Signal Confidence
Combines multiple signals (length, agreement, density, axes, quoted fraction)
into a final uncertainty score.
"""
from dataclasses import dataclass
from biaslab.config import CONFIDENCE_ZONES, HALF_CONFIDENCE

@dataclass
class ConfidenceResult:
    score: float
    zone: str
    zone_message: str
    breakdown: dict

def compute_confidence(n_words: int, n_sentences: int, sentence_agreement: float, ev_density: float, q_frac: float, hot_axes: int) -> ConfidenceResult:
    """Computes a 0.0-1.0 confidence score weighted from multiple robust factors."""
    sig_len = min(1.0, (n_words / (n_words + HALF_CONFIDENCE)))
    sig_agr = min(1.0, sentence_agreement)
    sig_ev = min(1.0, (ev_density / (ev_density + 8.0)))
    sig_axs = {0: 0.15, 1: 0.35, 2: 0.55, 3: 0.8}.get(min(hot_axes, 3), 1.0)
    sig_qt = max(0.0, 1.0 - q_frac * 1.5)

    score = max(0.0, min(1.0, sum([
        0.20 * sig_len, 0.25 * sig_agr, 0.20 * sig_ev, 0.20 * sig_axs, 0.15 * sig_qt
    ])))

    zone, msg = next(((z, m) for c, z, m in CONFIDENCE_ZONES if score < c), ("HIGH", "Analysis has strong evidential support"))
    
    return ConfidenceResult(round(score, 3), zone, msg, {
        "length": round(sig_len,3), "sentence_agreement": round(sig_agr,3), "evidence_density": round(sig_ev,3),
        "axis_agreement": round(sig_axs,3), "quote_adjustment": round(sig_qt,3)
    })
