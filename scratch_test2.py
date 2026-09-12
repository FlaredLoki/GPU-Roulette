"""Quick smoke test: prefix sum fix_suggestions diagnostic."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from src.predictor import GPUGatePredictor
from src.diagnostics import format_diagnostic
import json

pred = GPUGatePredictor()

code = """
void prefix(int N, double *A) {
    for (int i = 1; i < N; i++) {
        A[i] = A[i] + A[i-1];
    }
}
"""
results = pred.predict(code)
for r in results:
    print("=== format_diagnostic output ===")
    print(format_diagnostic(r, filename="prefix.c"))
    print()
    print("=== cli.py JSON shape ===")
    d = {
        "line": r.loop_line,
        "verdict": r.verdict,
        "fix_suggestions": [(f.get("problem", "") + " -> " + f.get("fix", "")) if isinstance(f, dict) else str(f) for f in r.fix_suggestions],
    }
    print(json.dumps(d, indent=2))
    print()
    print("=== Raw fix_suggestions dicts ===")
    for f in r.fix_suggestions:
        print(f)
