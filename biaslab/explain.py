"""
explain.py — Explanation Layer
Maps numeric values to strings and suggestions.
"""
from biaslab.config import VERDICT_THRESHOLDS

def build_verdict(score: float, zone: str = "HIGH") -> tuple:
    for c, l, m in VERDICT_THRESHOLDS:
        if score < c:
            if zone == "LOW": l += " (low confidence)"; m += " Note: low confidence — treat as suggestive."
            elif zone == "MEDIUM": m += " Confidence is moderate — consider reviewing."
            return l, m
    return "UNKNOWN", ""

def describe_lean(lean: float) -> str:
    if lean <= -0.6: return "Strong LEFT lean"
    if lean <= -0.2: return "Moderate LEFT lean"
    if lean < 0.2: return "Roughly centered"
    if lean < 0.6: return "Moderate RIGHT lean"
    return "Strong RIGHT lean"

def describe_confidence(conf: float, zone: str = "") -> str:
    return "LOW — insufficient evidence" if zone == "LOW" or conf < 0.4 else "MEDIUM — possible bias" if zone == "MEDIUM" or conf < 0.7 else "HIGH — strong evidential support"

def build_suggestions(r: dict) -> list:
    tips, d, ctx, z = [], r["details"], r.get("context_summary", {}), r.get("confidence_zone", "HIGH")
    if z == "LOW": tips.append("⚠ Confidence is LOW. Treat scores as suggestive.")
    if d["Loaded Language"]["score"] >= 40: tips.append("Replace dramatic words with plain facts.")
    if d["Sentiment Imbalance"]["score"] >= 40: tips.append("Add a counter-perspective to balance the tone.")
    if d["Subjectivity"]["score"] >= 40: tips.append("Drop opinion markers ('obviously').")
    if d["Source Opacity"]["score"] >= 50: tips.append("Attribute quotes to named sources instead of 'experts'.")
    if d["Sensational Framing"]["score"] >= 40: tips.append("Use neutral verbs instead of 'slammed' or 'attacked'.")
    if d["Political Slant"]["score"] >= 40: tips.append("Limit political buzzwords.")
    if r.get("clickbait_gap", 0) >= 40: tips.append("Headline is much more dramatic than the body.")
    if ctx.get("hits_excluded_quoted", 0) > 3: tips.append(f"{ctx['hits_excluded_quoted']} loaded words in quotes were excluded.")
    if ctx.get("hits_negated", 0) > 1: tips.append(f"{ctx['hits_negated']} negated words were downweighted.")
    return tips or ["Looks fairly balanced. Still verify facts independently."]

def build_evidence_summary(r: dict) -> str:
    lines, ctx = ["CONTEXTUAL ANALYSIS SUMMARY", "="*40, ""], r.get("context_summary", {})
    if r.get("confidence_zone") == "LOW": lines.extend(["⚠ LOW CONFIDENCE — treat scores as suggestive", ""])
    lines.extend([f"  {k.replace('_',' ').title()[:20]:20s} : {v}" for k, v in ctx.items()])
    if sents := r.get("sentence_data", []):
        lines.extend(["\nPER-SENTENCE EVIDENCE", "-"*40])
        for s in sents:
            tags = [t for t, c in [("HEADLINE", s["is_headline"]), ("QUOTED", s["is_quoted"]), ("ATTRIBUTED", s["is_attributed"])] if c] or ["AUTHOR"]
            lines.extend([f"\n  [{s['index']+1}] {' | '.join(tags)} (wt={s['context_weight']:.2f}, sig={s['total_signal']})", f"      \"{s['text'][:77]}...\""])
            if s["found_phrases"]: lines.append(f"      Found: {', '.join(s['found_phrases'][:8])}")
            if s["negated_phrases"]: lines.append(f"      Negated: {', '.join(s['negated_phrases'])}")
            if s["excluded_quoted"]: lines.append(f"      Excluded (quoted): {', '.join(s['excluded_quoted'])}")
    return "\n".join(lines)
