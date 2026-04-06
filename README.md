# sentvols — Financial Sentiment Analysis Toolkit

> **ECN 6578A — Projet de session, Fintech**  
> Analyse de sentiment sur les nouvelles financières pour alimenter un discriminateur de texte et des modèles de volatilité implicite.

---

## Overview

**sentvols** is a modular Python library for financial-news sentiment analysis.  
It combines rule-based and LLM-based annotation into a single, pluggable pipeline — from raw headlines to signed sentiment scores ready for downstream ML models (volatility prediction, factor models, portfolio signal generation).

```
Raw headline
    │
    ▼
[Optional] FinancialTextNormalizer   ← LLM rewriting (sarcasm, deep negation)
    │
    ▼
FinancialVADERAnnotator              ← VADER + Loughran-McDonald + phrase table
    │                                   + phrase negation guard + positional decay
    ▼
score ∈ [−1, 1]   label ∈ {positif, neutre, négatif}
```

Or with the façade:

```python
from sentvols.utils import Annotator

ann = Annotator()                             # zero-dependency default
ann.score("Company beats earnings estimates") # → +0.67

ann = Annotator(normalizer="llama_cpp", model_path="/tmp/model.gguf")
ann.score("Oh great, another record quarter — record losses, that is.")  # → −0.36
```

---

## Features

| Feature | Description |
|---|---|
| **VADER + LM dictionary** | 3 908 lexicon entries from the Loughran-McDonald Master Dictionary (1993–2025), merged into VADER's scorer |
| **29-phrase financial table** | Multi-word patterns (`"profit warning"`, `"chapter 11"`, `"earnings beat"`, …) with tunable phrase weight |
| **Phrase negation guard** | Detects negators within a 6-token window; flips phrase signal and blanks constituent tokens from VADER to eliminate double-counting |
| **Positional decay scoring** | `score_article()` splits on sentence boundaries and applies geometric decay so the lead sentence carries the most weight |
| **LLM normalizer** | `FinancialTextNormalizer` with pluggable backends: OpenAI, Anthropic, Transformers, Ollama, LlamaCpp — rewrites sarcasm and long-range negation before VADER sees the text |
| **Pure LLM annotator** | `FinancialLLMAnnotator` — delegates scoring directly to any LLM backend; useful as input features for text discriminators |
| **Pipeline façade** | `Annotator` — accepts string shortcuts or instances; sensible defaults; handles the full VADER + normalizer chain in one object |
| **Custom lexicon API** | `add_words()`, `add_phrases()`, `remove_words()`, `lexicon_snapshot` |
| **Batch + parallel** | `score_batch()`, `annotate_batch()`, `explain_batch()` with `ThreadPoolExecutor` |
| **Explainability** | `explain()`, `explain_to_dataframe()` — per-token and per-phrase contribution breakdown |
| **DataFrame integration** | `annotate_news(df, annotator, normalizer=)` — scores a news DataFrame with optional LLM normalisation |

---

## Project structure

```
sentvols/
├── core/
│   ├── annotators.py      # FinancialVADERAnnotator, FinancialLLMAnnotator, Annotator
│   ├── normalizers.py     # FinancialTextNormalizer + 6 LLM backends
│   ├── models.py          # Downstream ML models (LightGBM discriminator, …)
│   ├── plots.py           # Sentiment waterfall, distribution, time-series charts
│   ├── explainers.py      # SHAP-based explanations for ML models
│   ├── utils.py           # annotate_news(), DataFrame helpers
│   └── exports.py         # @registration decorator for the public API
├── utils/                 # Public namespace (sentvols.utils.*)
├── explainers/            # Public namespace (sentvols.explainers.*)
├── models/                # Public namespace (sentvols.models.*)
├── plots/                 # Public namespace (sentvols.plots.*)
└── internals/
    └── Loughran-McDonald_MasterDictionary_1993-2025.csv

tests/
├── test_annotators.py     # 118 tests
├── test_normalizers.py    # 42 tests
├── test_models.py
├── test_plots.py
├── test_explainers.py
└── test_utils.py

notebooks/
└── sentivols.ipynb        # End-to-end demo

rapports/
└── rapport.tex            # LaTeX report
```

---

## Installation

**Requirements:** Python ≥ 3.10, conda base environment (includes `vaderSentiment`, `polars`, `pandas`, `lightgbm`).

```bash
# Clone and enter the project
cd "projet"

# Install core dependencies (if not already present)
pip install vaderSentiment pandas polars lightgbm matplotlib seaborn

# Optional — local LLM inference (llama.cpp / Ollama)
python -m venv .venv
.venv/bin/pip install llama-cpp-python ollama
```

The Loughran-McDonald dictionary is bundled at `sentvols/internals/` — no external download needed.

---

## Quick start

### Plain VADER annotator (zero dependencies beyond vaderSentiment)

```python
from sentvols.utils import FinancialVADERAnnotator

ann = FinancialVADERAnnotator()

ann.score("Company beats earnings estimates and raises guidance.")   # +0.67  POS
ann.score("The firm filed for Chapter 11 bankruptcy.")               # -1.00  NEG
ann.score("Analysts confirmed the firm did not issue a profit warning.")  # +0.95  POS

# Batch
ann.score_batch(headlines, workers=4)
ann.annotate_batch(headlines)          # → [{"score": float, "label": str}, …]

# Long articles with positional decay
ann.score_article(article_text, decay=0.9)

# Custom vocabulary
ann.add_words({"synergies": 2.5, "disintermediation": -2.0})
ann.add_phrases({"shareholder value creation": 2.0})

# Explainability
exp = ann.explain("Record earnings beat with profit warning for Q4")
# {"base_vader_score": …, "word_hits": […], "phrase_hits": […], "final_score": …}

df = ann.explain_to_dataframe(text)    # pandas DataFrame, sorted by |valence|
```

### With LLM normalisation (handles sarcasm, long-range negation)

```python
from sentvols.utils import Annotator

# Local inference — no API key, no server
ann = Annotator(
    normalizer="llama_cpp",
    model_path="/tmp/qwen2.5-0.5b-instruct-q4_k_m.gguf",
)

ann.score(
    "Oh great, another record quarter — record losses, that is."
)  # → -0.36  NEG  (plain VADER gives +0.74)

ann.score(
    "Analysts do not expect the firm to file for bankruptcy protection."
)  # → +0.72  POS  (plain VADER gives -0.54)
```

### Pure LLM annotator

```python
from sentvols.utils import FinancialLLMAnnotator
from sentvols.core.normalizers import OllamaBackend

backend = OllamaBackend(model="qwen2.5:1.5b")
ann = FinancialLLMAnnotator(backend)

ann.score("Company posts record losses amid guidance cut")   # → -0.8
ann.annotate_batch(headlines, workers=2)
```

### DataFrame workflow

```python
import polars as pl
from sentvols.utils import Annotator
from sentvols.core.utils import annotate_news

ann = Annotator()
df = pl.read_csv("data/raw/headlines.csv")
df = annotate_news(df, ann.inner_annotator)
# Adds columns: score, label
```

---

## Annotator comparison (20-case benchmark)

| Pipeline | Accuracy | Best for |
|---|---|---|
| `FinancialVADERAnnotator` | **16/20 (80%)** | Clean headlines, zero latency |
| `Annotator(normalizer="llama_cpp")` | **17/20 (85%)** | Sarcasm, long-range negation |
| `FinancialLLMAnnotator` (0.5B) | 4/20 — not viable at this size | Use ≥1.5B models |

Per-category breakdown:

| Category | VADER | Hybrid |
|---|---|---|
| Baseline (clear POS/NEG) | 6/6 ✓ | 5/6 |
| Explicit negation (phrase guard) | 3/3 ✓ | 3/3 ✓ |
| Long-range negation | 1/2 | **2/2 ✓** |
| Sarcasm | 1/2 | 1/2 |
| Buried negatives | 2/2 ✓ | 2/2 ✓ |
| Double negation | 1/1 ✓ | 1/1 ✓ |
| Mixed signals | 1/2 | **2/2 ✓** |

---

## LLM backends

| Backend | Class | Requires |
|---|---|---|
| OpenAI (gpt-4o, gpt-4o-mini) | `OpenAIBackend` | `openai` pip package + API key |
| Anthropic (claude-3-*) | `AnthropicBackend` | `anthropic` pip package + API key |
| o1 / o3 (reasoning) | `ReasoningBackend` | `openai` pip package + API key |
| HuggingFace (flan-t5, etc.) | `TransformersBackend` | `transformers`, `torch` |
| Ollama (local daemon) | `OllamaBackend` | `ollama` pip package + running daemon |
| llama.cpp (GGUF, no server) | `LlamaCppBackend` | `llama-cpp-python` |

All backends expose the same `NormalizerBackend` protocol: `model`, `reasoning_available`, `call(prompt) → (text, reasoning_trace)`.

---

## Running the tests

```bash
# Uses the base conda environment
python -m pytest tests/ -q
# 249 passed

# Verbose with coverage per file
python -m pytest tests/ -v --tb=short
```

---

## Lexicon details

The VADER scorer is patched with a financial-specific lexicon built from three sources, in priority order:

1. **Loughran-McDonald Master Dictionary** (1993–2025) — 3 908 entries covering Positive (+2.5), Negative (−2.5), Litigious (−1.0), Uncertainty (−0.5), Constraining (−0.3)
2. **financial-neutral override** — 25 VADER-mislabelled words (`"market"`, `"stock"`, `"option"`, …) forced to 0.0
3. **headline patch** — 30 newswire verbs absent from SEC-filing vocabulary (`"beats"`, `"misses"`, `"surges"`, `"downgraded"`, …)

Plus a 29-phrase context table for multi-word patterns, with a configurable `phrase_weight` (default 0.3) and a 6-token negation window.

---

## Course context

**ECN 6578A — Économétrie des marchés financiers**  
Université du Québec à Montréal (UQAM)  
Session H2025 — Projet Fintech (20% de la note finale)

The goal is to build a proof-of-concept application demonstrating how natural language processing can extract tradeable sentiment signals from financial news, bridging NLP and empirical asset pricing.
