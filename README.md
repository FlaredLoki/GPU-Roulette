# 🎰 GPU Roulette

**Should this loop run on the GPU — or is that a bet you'll lose?**

GPU Roulette is a static analysis tool that looks at a C/C++ loop and tells you, before you ever compile a CUDA kernel, whether offloading it to a GPU will actually pay off. It parses your code with Tree-sitter, extracts structural and memory-access features, and runs them through an XGBoost model trained on a **physics-informed simulation of real GPU offload economics** — not just a black-box classifier, but one grounded in actual hardware bottlenecks like PCIe bandwidth and kernel launch overhead.

Point it at a loop. It gives you a verdict, a confidence score, a SHAP-based explanation of *why*, and — if the loop is unprofitable — concrete suggestions for what to change.

---

## Why this exists

"Just throw it on the GPU" is bad advice more often than people admit. A loop with a tiny trip count, heavy memory transfer relative to compute, or dynamic memory allocation can be *slower* on a GPU than on the CPU once you account for the cost of moving data across the PCIe bus. Most developers find this out the hard way, after writing the kernel.

GPU Roulette catches it statically, in seconds, before a single line of GPU code gets written.

## How it works

```
C/C++ source
     │
     ▼
Tree-sitter AST parse  ──────────►  syntax-level blockers?
     │                              (malloc, free, exit, etc.)
     ▼                                       │
Feature extraction                           ▼
(trip count, stride pattern,           immediate BLOCKED verdict
 data reuse, memory footprint,         (skips the ML model entirely)
 branch/dependency analysis)
     │
     ▼
XGBoost classifier ────────► PROFITABLE / UNPROFITABLE / confidence-capped UNCERTAIN
     │
     ▼
SHAP explainer ────────► feature-by-feature breakdown of the verdict
```

The classifier itself isn't trained on arbitrary labels — it's trained on a simulated cost model that computes actual CPU-vs-GPU wall-clock time from three physical terms: GPU compute time, PCIe transfer time, and kernel launch overhead. A loop only gets labeled `PROFITABLE` in training if it would *genuinely* be faster end-to-end, including the cost of getting the data there and back. That's why the model correctly flags even a heavily compute-bound but astronomically large loop as unprofitable — the data transfer alone can dominate.

When the parser can't statically resolve something with confidence (a pointer-aliased array access, a loop with a data-dependent early exit via `break`/`continue`/`goto`), the tool doesn't guess with false precision — confidence is explicitly capped and the reason is surfaced in the explanation.

## What it catches

| Verdict | Meaning |
|---|---|
| 🟢 **PROFITABLE** | Compute intensity justifies the transfer cost — GPU offload wins |
| 🔴 **UNPROFITABLE** | Transfer/launch overhead outweighs the compute gain — stay on CPU |
| 🟡 **UNCERTAIN** (confidence-capped) | Static analysis genuinely can't resolve the bound or dependency — flagged honestly rather than guessed |
| ⛔ **BLOCKED** | Hard GPU-incompatible constructs detected (`malloc`, `free`, `exit`) — bypasses the ML model entirely at 100% confidence |

## Quickstart

```bash
python -m venv venv
# Windows
.\venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

pip install -r requirements.txt
```

### Web UI

Side-by-side C source, live verdict, and a full SHAP-style feature breakdown.

```bash
python web/server.py
```
Then open `http://127.0.0.1:5000`.

### CLI

Batch-analyze files or pipe in inline code — the same engine that powers the web UI.

```bash
# Single file
python -m src.cli analyze path/to/file.c

# Inline snippet
python -m src.cli analyze --code "for (int i=0;i<N;i++) { a[i] += b[i]; }"

# Batch mode with a JSON report
python -m src.cli batch src/*.c --report results.json
```

### Tests

```bash
pytest tests/
```

## Known limitations

Static analysis has hard edges — macro-expanded bounds, pointer-aliased loops, and data-dependent control flow are all real limits on what can be known before runtime. Rather than hide them, they're documented explicitly in [`KNOWN_LIMITATIONS.md`](./KNOWN_LIMITATIONS.md), including exactly which fallback heuristic kicks in for each case.

## Built for SegFault 2026

Built during [SegFault 2026](https://segfault.compilertech.org/), a hackathon on compilers and programming languages organized as part of the Innovations In Compiler Technology (IICT) workshop at IISc Bengaluru. This project directly addresses the official problem statement **"Parallelization Profitability Predictor for GPU."**

**Team roles:**

- **Pranav A** — static analysis engine: C parser (`code_parser.py`), 17-feature extractor (`feature_extractor.py`), roofline positioning (`roofline.py`), fix-suggestion engine (`fixits.py`), CLI (`cli.py`); also handled integration, debugging, and production hardening (thread-safety fix, O(N³) memory-footprint correction, dependency resolution, environment-agnostic packaging)
- **[Urvi U]** — ML pipeline: physics-informed synthetic training data (`synthetic_data.py`), XGBoost model and rule-based baseline (`model.py`, `baseline.py`), training script (`train.py`), SHAP-based explainer (`explainer.py`), profitability crossover analysis (`crossover.py`), compiler-style diagnostics (`diagnostics.py`), web UI (`web/`)
- **Both** — joint integration layer (`predictor.py`)

**Stack:** Python · Tree-sitter · XGBoost · SHAP · Flask
