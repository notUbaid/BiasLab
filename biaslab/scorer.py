"""
scorer.py — Stage C+D: Feature Scoring and Aggregation
Calculates per-axis and overall bias scores based on context-weighted features,
and computes final multi-signal confidence scores.
"""
from __future__ import annotations
from datetime import datetime
from typing import List
from biaslab.config import (APP_VERSION, RADAR_LABELS, HALF_SENSATIONAL, HALF_ABSOLUTE, HALF_FRAMING, HALF_SLANT, 
                            HALF_WEASEL, HALF_EXCLAIM, HALF_CAPS, HALF_SUPERLATIVE, HALF_CLICKBAIT, WEIGHT_NEGATED)
from biaslab.preprocessing import AnnotatedSentence, preprocess_article
from biaslab.features import SentenceFeatures, extract_all_features
from biaslab.confidence import compute_confidence

def _smooth(val: float, half: float) -> float: return 100.0 * val / (val + half) if val > 0 else 0.0
def _density(count: float, n_words: int) -> float: return (count / max(n_words, 1)) * 1000.0

def _get_axis(features: List[SentenceFeatures], axis_name: str) -> tuple[float, List[str]]:
    """Helper to extract weighted total and found phrases for an axis."""
    wt = sum(f.axis_data[axis_name]["raw"] * f.context_weight for f in features)
    found = [p for f in features for p in f.axis_data[axis_name]["found"]]
    return wt, found

def _score_axes(f: List[SentenceFeatures], w_tot: int, s: dict) -> list[dict]:
    """Calculate the 6 radar axes utilizing context weights."""
    n_neg = sum(f_.n_negated_hits for f_ in f)
    
    # 1. Loaded Language
    w_sens, f_sens = _get_axis(f, "sensational")
    w_sens += n_neg * WEIGHT_NEGATED * 0.3
    d_sens = _density(w_sens, w_tot)
    d_excl = _density(sum(x.exclamation_count for x in f), w_tot)
    d_caps = _density(sum(x.caps_word_count for x in f), w_tot)
    score_idx1 = 0.6 * _smooth(d_sens, HALF_SENSATIONAL) + 0.25 * _smooth(d_excl, HALF_EXCLAIM) + 0.15 * _smooth(d_caps, HALF_CAPS)
    
    # 2. Sentiment Imbalance
    w_pos, f_pos = _get_axis(f, "positive")
    w_neg, f_neg = _get_axis(f, "negative")
    tot_sent = w_pos + w_neg
    imb = abs(w_pos - w_neg) / max(tot_sent, 1)
    score_idx2 = (imb * _smooth(_density(tot_sent, w_tot), 28.0)) if tot_sent >= 4 else 0.0
    f_sent = sorted(set(f_pos if w_pos >= w_neg else f_neg))
    
    # 3. Subjectivity
    w_abs, f_abs = _get_axis(f, "absolute")
    w_hed, _ = _get_axis(f, "hedge")
    score_idx3 = max(0.0, _smooth(_density(w_abs, w_tot), HALF_ABSOLUTE) - min(15.0, _density(w_hed, w_tot)*2.0)) if (w_abs>=2 or _density(w_abs,w_tot)>=3) else 0.0
    
    # 4. Source Opacity
    w_wea, f_wea = _get_axis(f, "weasel")
    w_cit, _ = _get_axis(f, "citation")
    rat_score = 100 * w_wea / (w_wea + w_cit) if (w_wea + w_cit) > 0 else 20.0
    score_idx4 = max(0.0, 0.55 * rat_score + 0.45 * _smooth(_density(w_wea, w_tot), HALF_WEASEL) - min(15.0, _density(s.get("quoted_segments",0), w_tot)*5.0))
    
    # 5. Framing
    w_frm, f_frm = _get_axis(f, "framing")
    w_sup, _ = _get_axis(f, "superlative")
    score_idx5 = 0.7 * _smooth(_density(w_frm, w_tot), HALF_FRAMING) + 0.3 * _smooth(_density(w_sup, w_tot), HALF_SUPERLATIVE)
    
    # 6. Slant
    w_lft, f_lft = _get_axis(f, "left")
    w_rgt, f_rgt = _get_axis(f, "right")
    tot_slt = w_lft + w_rgt
    score_idx6 = _smooth(_density(tot_slt, w_tot), HALF_SLANT) if tot_slt > 0 else 0.0
    lean = (w_rgt - w_lft) / max(tot_slt, 1)
    
    return [
        {"score": score_idx1, "found": sorted(set(f_sens)), "count": int(w_sens), "sub": {"note":"Context-weighted counts"}},
        {"score": score_idx2, "found": f_sent, "count": int(tot_sent), "sub": {"sentiment_density": round(_density(tot_sent, w_tot),2)}},
        {"score": score_idx3, "found": sorted(set(f_abs)), "count": int(w_abs), "sub": {"hedge_relief": round(min(15.0, _density(w_hed, w_tot)*2.0),2)}},
        {"score": score_idx4, "found": sorted(set(f_wea)), "count": int(w_wea), "sub": {"weasel_ratio": round(rat_score,2)}},
        {"score": score_idx5, "found": sorted(set(f_frm)), "count": int(w_frm), "sub": {"framing_density": round(_density(w_frm, w_tot),2)}},
        {"score": score_idx6, "found": sorted(set(f_rgt if w_rgt >= w_lft else f_lft)), "count": int(tot_slt), "lean": lean, "sub": {"density": round(_density(tot_slt, w_tot),2)}},
    ]

def analyze_article(title: str, body: str) -> dict:
    """End-to-end preprocessing, feature extraction, scoring, and analysis generation."""
    sents, stats = preprocess_article(title, body)
    feats = extract_all_features(sents)
    w_tot = stats["n_words"]
    
    axes_res = _score_axes(feats, w_tot, stats)
    radar = [round(a["score"], 1) for a in axes_res]
    
    # Overall Score Aggregation
    s_mean = sum(radar) / len(radar)
    hot = sum(1 for v in radar if v >= 30)
    overall = (0.5 * s_mean + 0.5 * sum(sorted(radar)[-3:]) / 3.0) if hot >= 3 else s_mean
    
    sig_cnt = sum(1 for f in feats if not f.is_quoted and sum(f.axis_data[k]["raw"] for k in f.axis_data) > 0)
    sig_breadth = sig_cnt / max(len(feats), 1)
    overall += (5.0 if hot >= 4 and sig_breadth >= 0.4 else 3.0 if hot >= 5 and sig_breadth >= 0.5 else 0)
    overall = min(100.0, overall)
    
    # Clickbait Gap
    hd_f = [f for f in feats if f.is_headline]
    bd_f = [f for f in feats if not f.is_headline]
    hd_d = sum(sum(f.axis_data[k]["raw"] for k in ["sensational","framing","superlative","absolute"]) for f in hd_f)
    bd_d = sum(sum(f.axis_data[k]["raw"] for k in ["sensational","framing","superlative","absolute"]) for f in bd_f)
    den_hd = _density(hd_d, sum(f.word_count for f in hd_f) or 1)
    den_bd = _density(bd_d, sum(f.word_count for f in bd_f) or 1)
    cb_score = _smooth(max(0.0, den_hd - den_bd), HALF_CLICKBAIT)
    
    # Confidence
    auth_sents = [f for f in feats if not f.is_quoted and f.context_weight >= 0.5]
    sent_agrmt = sum(1 for f in auth_sents if sum(f.axis_data[k]["raw"] for k in f.axis_data) > 0) / max(len(auth_sents), 1)
    ev_density = _density(sum(sum(f.axis_data[k]["raw"] for k in f.axis_data) for f in feats), w_tot)
    conf = compute_confidence(w_tot, len(sents), sent_agrmt, ev_density, stats["quoted_sentence_fraction"], hot)
    
    # Final Report
    return {
        "version": APP_VERSION, "title": title.strip(), "when": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_words": w_tot, "confidence": conf.score, "confidence_zone": conf.zone,
        "confidence_message": conf.zone_message, "confidence_breakdown": conf.breakdown,
        "radar_labels": RADAR_LABELS, "radar_values": radar, "overall_score": round(overall, 1),
        "political_lean": round(axes_res[5].get("lean", 0.0), 2),
        "clickbait_gap": round(cb_score,1), "title_drama": round(den_hd,2), "body_drama": round(den_bd,2),
        "stats": stats,
        "context_summary": {
            "sentences_analyzed": len(sents), "sentences_quoted": sum(1 for f in feats if f.is_quoted),
            "sentences_attributed": sum(1 for f in feats if f.is_attributed),
            "hits_excluded_quoted": sum(len(f.excluded_quoted) for f in feats),
            "hits_negated": sum(f.n_negated_hits for f in feats),
            "sentence_agreement": round(sent_agrmt, 2), "evidence_density": round(ev_density, 2), "signal_breadth": round(sig_breadth, 2),
        },
        "details": dict(zip([l.replace("\n"," ") for l in RADAR_LABELS], axes_res)),
        "sentence_data": [{"index": f.sentence_index, "text": sents[f.sentence_index].text if f.sentence_index < len(sents) else "",
                           "is_headline": f.is_headline, "is_quoted": f.is_quoted, "is_attributed": f.is_attributed,
                           "context_weight": f.context_weight, "total_signal": sum(f.axis_data[k]["raw"] for k in f.axis_data),
                           "negated_hits": f.n_negated_hits, "negated_phrases": f.negated_phrases, "excluded_quoted": f.excluded_quoted,
                           "found_phrases": [p for k in f.axis_data for p in f.axis_data[k]["found"]]} for f in feats]
    }
