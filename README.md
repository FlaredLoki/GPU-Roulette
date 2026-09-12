# GPU Roulette
**GPU Parallelization Profitability Predictor**

A static analysis and ML-based tool that analyzes C/C++ code to predict if a loop is profitable to offload to a GPU. It uses Tree-sitter for robust syntax tree parsing and an XGBoost model for classification and hardware-aware heuristics.

## Setup

```bash
python -m venv venv
# Windows
.\venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

pip install -r requirements.txt
```

## Running the Web UI

The Web UI provides a side-by-side view of the C code, diagnostic predictions, and visual explanations (SHAP).

```bash
python web/server.py
```
Then navigate to `http://127.0.0.1:5000` in your browser.

## Running the CLI

The Command Line Interface lets you batch analyze C files.

```bash
python -m src.cli analyze path/to/file.c
```

## Running Tests

To run the test suite or any of the `run_batch*.py` testing scripts, `pytest` and `requests` are required (both are included in `requirements.txt`).

```bash
pytest tests/
```

## Known Limitations
Please see `KNOWN_LIMITATIONS.md` for a documented list of static analysis boundaries and fallback behaviors.
