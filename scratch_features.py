"""Dump full features_dict for the Jacobi stencil."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from src.predictor import GPUGatePredictor
from src.diagnostics import format_diagnostic
import json

with open('test_profitable.c', 'r') as f:
    code = f.read()

pred = GPUGatePredictor()
results = pred.predict(code)

for r in results:
    print("=== VERDICT ===")
    print(f"{r.verdict} ({r.confidence*100:.1f}% confidence)")
    print()
    print("=== FEATURES DICT ===")
    for k, v in r.features.items():
        print(f"  {k:30s} = {v}")
    print()
    print("=== SHAP VALUES ===")
    for k, v in r.shap_values.items():
        print(f"  {k:30s} = {v:+.4f}")
    print()
    print("=== ROOFLINE ===")
    for k, v in r.roofline_position.items():
        print(f"  {k:30s} = {v}")
    print()
    print("=== BLOCKERS ===")
    print(r.blockers)
    print()
    print("=== FIX SUGGESTIONS ===")
    for f_s in r.fix_suggestions:
        print(f"  {f_s}")
    print()
    print("=== DIAGNOSTIC ===")
    print(format_diagnostic(r, filename="test_profitable.c"))
