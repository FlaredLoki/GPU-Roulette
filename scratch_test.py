"""Quick smoke test: printf blocker diagnostic + JSON output."""
from src.predictor import GPUGatePredictor
from src.diagnostics import format_diagnostic, format_batch_summary
import json

code = """
void foo(int N) {
    for (int i = 0; i < N; i++) {
        printf("%d", i);
    }
}
"""
pred = GPUGatePredictor()
results = pred.predict(code)

print("=== format_diagnostic output ===")
for r in results:
    print(format_diagnostic(r, filename="test.c"))
    print()

print("=== format_batch_summary output ===")
print(format_batch_summary(results, "test.c"))
print()

# Also test the JSON path (cli.py shape)
print("=== cli.py JSON shape ===")
for r in results:
    d = {
        "line": r.loop_line,
        "verdict": r.verdict,
        "confidence": r.confidence,
        "crossover_n": r.crossover_n,
        "blockers": [b.get("reason", str(b)) if isinstance(b, dict) else str(b) for b in r.blockers],
        "fix_suggestions": [(f.get("problem", "") + " -> " + f.get("fix", "")) if isinstance(f, dict) else str(f) for f in r.fix_suggestions],
    }
    print(json.dumps(d, indent=2))

# Now test an UNPROFITABLE case to see fix_suggestions actually render
print()
print("=== UNPROFITABLE case (prefix sum, has fix_suggestions) ===")
code2 = """
void prefix(int N, double *A) {
    for (int i = 1; i < N; i++) {
        A[i] = A[i] + A[i-1];
    }
}
"""
results2 = pred.predict(code2)
for r in results2:
    print(format_diagnostic(r, filename="prefix.c"))
    print()
    d = {
        "line": r.loop_line,
        "verdict": r.verdict,
        "fix_suggestions": [(f.get("problem", "") + " -> " + f.get("fix", "")) if isinstance(f, dict) else str(f) for f in r.fix_suggestions],
    }
    print(json.dumps(d, indent=2))
