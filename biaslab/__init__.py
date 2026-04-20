"""BiasLab v3 — Hybrid Contextual News-Bias Analyzer."""
__version__ = "3.0"
APP_VERSION = __version__
from biaslab.scorer import analyze_article  # noqa: F401, E402
__all__ = ["analyze_article", "APP_VERSION"]
