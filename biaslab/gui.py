"""
gui.py — Tkinter GUI
"""
import os, re, traceback, tkinter as tk
from math import pi
from tkinter import ttk, messagebox, scrolledtext, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from biaslab.config import APP_VERSION, SAMPLE_TITLE, SAMPLE_BODY, SAMPLE_NEUTRAL_TITLE, SAMPLE_NEUTRAL_BODY
from biaslab.scorer import analyze_article
from biaslab.explain import build_verdict, describe_lean, describe_confidence, build_suggestions, build_evidence_summary
from biaslab.export import save_session, export_report_pdf

class BiasLabApp:
    def __init__(self, root: tk.Tk):
        self.root, self.curr = root, None
        self.root.title(f"BiasLab v{APP_VERSION}"); self.root.geometry("1320x860"); self.root.configure(bg="#0f172a")
        self._build_ui()

    def _build_ui(self):
        ttk.Style().theme_use("clam")
        man = tk.Frame(self.root, bg="#0f172a"); man.pack(fill="both", expand=True, padx=16, pady=12)
        man.grid_columnconfigure(0, weight=1); man.grid_columnconfigure(1, weight=1); man.grid_rowconfigure(0, weight=1)
        
        # Left Panel
        pL = tk.Frame(man, bg="#1e293b"); pL.grid(row=0, column=0, sticky="nsew", padx=8)
        tk.Label(pL, text="1. Paste Article", fg="white", bg="#1e293b").pack(pady=4)
        self.eTitle = tk.Entry(pL, bg="#172033", fg="white"); self.eTitle.pack(fill="x", padx=14, pady=4)
        self.tBody = scrolledtext.ScrolledText(pL, bg="#172033", fg="white", height=20); self.tBody.pack(fill="both", expand=True, padx=14, pady=4)
        
        br = tk.Frame(pL, bg="#1e293b"); br.pack(fill="x", padx=14, pady=14)
        ttk.Button(br, text="Analyze", command=self.do_analyze).pack(side="left")
        ttk.Button(br, text="Bias Sample", command=lambda: self.load(SAMPLE_TITLE, SAMPLE_BODY)).pack(side="left", padx=2)
        ttk.Button(br, text="Neutral Sample", command=lambda: self.load(SAMPLE_NEUTRAL_TITLE, SAMPLE_NEUTRAL_BODY)).pack(side="left")
        ttk.Button(br, text="Clear", command=self.clear).pack(side="left", padx=2)
        self.bPdf = ttk.Button(br, text="Export PDF", command=self.do_pdf, state="disabled"); self.bPdf.pack(side="right")
        self.bCsv = ttk.Button(br, text="Save CSV", command=self.do_csv, state="disabled"); self.bCsv.pack(side="right", padx=2)
        
        self.stat = tk.StringVar(value="Ready."); tk.Label(pL, textvariable=self.stat, fg="#94a3b8", bg="#1e293b").pack(fill="x", padx=14)

        # Right Panel
        pR = tk.Frame(man, bg="#1e293b"); pR.grid(row=0, column=1, sticky="nsew", padx=8)
        self.tabs = ttk.Notebook(pR); self.tabs.pack(fill="both", expand=True, padx=10, pady=10)
        self.tSum = tk.Frame(self.tabs, bg="#1e293b"); self.tabs.add(self.tSum, text="Summary")
        self.tRad = tk.Frame(self.tabs, bg="#1e293b"); self.tabs.add(self.tRad, text="Radar")
        self.tEvd = scrolledtext.ScrolledText(self.tabs, bg="#172033", fg="white"); self.tabs.add(self.tEvd, text="Evidence")
        self.tTip = scrolledtext.ScrolledText(self.tabs, bg="#172033", fg="white"); self.tabs.add(self.tTip, text="Tips")

    def load(self, t, b): self.eTitle.delete(0, "end"); self.eTitle.insert(0, t); self.tBody.delete("1.0", "end"); self.tBody.insert("1.0", b)
    def clear(self): self.curr = None; self.load("", ""); self.bPdf.config(state="disabled"); self.bCsv.config(state="disabled")

    def do_analyze(self):
        t, b = self.eTitle.get().strip(), self.tBody.get("1.0", "end").strip()
        if len(b) < 50: return messagebox.showwarning("Error", "Need >50 chars")
        try:
            self.curr = analyze_article(t, b)
            self.bPdf.config(state="normal"); self.bCsv.config(state="normal")
            self._render(self.curr)
        except Exception as e: traceback.print_exc(); messagebox.showerror("Err", str(e))

    def do_csv(self): save_session(self.curr); messagebox.showinfo("Saved", "CSV Saved")
    def do_pdf(self):
        if not self.curr: return
        p = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF", "*.pdf")])
        if p: export_report_pdf(self.curr, p); messagebox.showinfo("Saved", f"PDF saved: {p}")

    def _render(self, r: dict):
        [w.destroy() for w in self.tSum.winfo_children()]
        tk.Label(self.tSum, text=f"Overall: {r['overall_score']:.1f}/100", fg="#4ade80", bg="#1e293b", font=("Segoe UI", 24)).pack(pady=20)
        
        self.tEvd.config(state="normal"); self.tEvd.delete("1.0", "end"); self.tEvd.insert("1.0", build_evidence_summary(r)); self.tEvd.config(state="disabled")
        self.tTip.config(state="normal"); self.tTip.delete("1.0", "end"); self.tTip.insert("1.0", "\n".join(build_suggestions(r))); self.tTip.config(state="disabled")
        
        [w.destroy() for w in self.tRad.winfo_children()]
        fig = plt.figure(figsize=(4, 4), facecolor="#1e293b"); ax = fig.add_subplot(111, polar=True, facecolor="#1e293b")
        v, l = r["radar_values"], r["radar_labels"]
        a = [i/len(v)*2*pi for i in range(len(v))]
        ax.plot(a+[a[0]], v+[v[0]], color="#38bdf8", lw=2); ax.fill(a+[a[0]], v+[v[0]], color="#38bdf8", alpha=0.3)
        ax.set_xticks(a); ax.set_xticklabels([x.replace("\n", " ") for x in l], color="white", fontsize=8)
        canv = FigureCanvasTkAgg(fig, master=self.tRad); canv.draw(); canv.get_tk_widget().pack(fill="both")

def main(): root = tk.Tk(); app = BiasLabApp(root); root.mainloop()
