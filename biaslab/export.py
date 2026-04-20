"""
export.py — CSV & PDF Export
"""
import csv, os, re
from math import pi
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from biaslab.config import APP_VERSION, CSV_FILENAME
from biaslab.explain import build_verdict, describe_lean, describe_confidence

CSV_COLUMNS = ["when", "title", "n_words", "confidence", "confidence_zone", "overall_score", "verdict", "political_lean", "clickbait_gap", "loaded", "sentiment", "subjectivity", "source_opacity", "framing", "slant", "sentences_analyzed", "hits_excluded_quoted", "hits_negated"]

def save_session(r: dict):
    if not os.path.exists(CSV_FILENAME) or os.path.getsize(CSV_FILENAME) == 0:
        with open(CSV_FILENAME, "w", newline="", encoding="utf-8") as f: csv.writer(f).writerow(CSV_COLUMNS)
    v_lbl, _ = build_verdict(r["overall_score"], r.get("confidence_zone", "HIGH"))
    row = [r["when"], r["title"][:80], r["n_words"], r["confidence"], r.get("confidence_zone", ""), r["overall_score"], v_lbl, r["political_lean"], r["clickbait_gap"], r["details"]["Loaded Language "]["score"], r["details"]["Sentiment Imbalance "]["score"], r["details"]["Subjectivity "]["score"], r["details"]["Source Opacity "]["score"], r["details"]["Sensational Framing "]["score"], r["details"]["Political Slant "]["score"], r.get("context_summary", {}).get("sentences_analyzed", 0), r.get("context_summary", {}).get("hits_excluded_quoted", 0), r.get("context_summary", {}).get("hits_negated", 0)]
    with open(CSV_FILENAME, "a", newline="", encoding="utf-8") as f: csv.writer(f).writerow(row)

def export_report_pdf(r: dict, path: str):
    cz, v_lbl, v_msg = r.get("confidence_zone", "HIGH"), *build_verdict(r["overall_score"], r.get("confidence_zone", "HIGH"))
    with PdfPages(path) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69), dpi=120, facecolor="#ffffff")
        gs = fig.add_gridspec(5, 2, height_ratios=[1, 0.35, 0.55, 1.25, 1.25], hspace=0.55, wspace=0.25, left=0.07, right=0.95, top=0.95, bottom=0.05)
        
        ax = fig.add_subplot(gs[0, 0]); ax.axis("off")
        ax.text(0, 1.0, "BiasLab v3", fontsize=20, fontweight="bold", color="#2563eb", transform=ax.transAxes)
        ax.text(0, 0.86, f"Summary: {r['title'][:72]}...", fontsize=10, transform=ax.transAxes)
        ax.text(0.02, 0.28, f"{r['overall_score']:.0f} / 100", fontsize=24, fontweight="bold", transform=ax.transAxes)
        
        ax_r = fig.add_subplot(gs[0, 1], polar=True); vals, lbls = r["radar_values"], r["radar_labels"]
        angs = [i/len(vals) * 2 * pi for i in range(len(vals))]
        ax_r.plot(angs+[angs[0]], vals+[vals[0]], color="#2563eb", linewidth=1.8)
        ax_r.fill(angs+[angs[0]], vals+[vals[0]], color="#2563eb", alpha=0.22)
        ax_r.set_xticks(angs); ax_r.set_xticklabels([l.replace("\n"," ") for l in lbls], fontsize=7)
        ax_r.set_yticks([25, 50, 75]); ax_r.set_ylim(0, 100)
        
        ax_b = fig.add_subplot(gs[3, :]); ax_b.axis("off")
        ax_b.barh(list(range(len(lbls)))[::-1], [r["details"][l.replace("\n"," ")]["score"] for l in lbls], color="#2563eb")
        
        pdf.savefig(fig, facecolor="#ffffff"); plt.close(fig)
