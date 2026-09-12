"""Find which feature(s) flip the model from PROFITABLE to UNPROFITABLE."""
import sys, io, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.append('.')

import pickle
from src.feature_extractor import FEATURE_NAMES

model = pickle.load(open('models/gpugate_model.pkl', 'rb'))

handcrafted = {
    'trip_count_log': 20.0, 'nest_depth': 1.0, 'inner_loop_count': 1.0,
    'total_ops': 5.0, 'has_float_ops': 1.0, 'comp_intensity': 1.0,
    'mem_reads': 5.0, 'mem_writes': 1.0, 'unique_arrays': 2.0,
    'stride_one_ratio': 0.85, 'has_indirect_access': 0.0,
    'data_reuse_score': 0.6, 'estimated_data_bytes_log': 22.0,
    'transfer_penalty': -1.0, 'branch_count': 0.0,
    'has_reduction': 0.0, 'has_loop_carried_dep': 0.0,
}

actual = {
    'trip_count_log': 19.997, 'nest_depth': 1.0, 'inner_loop_count': 1.0,
    'total_ops': 9.0, 'has_float_ops': 1.0, 'comp_intensity': 1.5,
    'mem_reads': 4.0, 'mem_writes': 1.0, 'unique_arrays': 2.0,
    'stride_one_ratio': 0.5, 'has_indirect_access': 0.0,
    'data_reuse_score': 0.6, 'estimated_data_bytes_log': 23.997,
    'transfer_penalty': 0.830, 'branch_count': 0.0,
    'has_reduction': 1.0, 'has_loop_carried_dep': 0.0,
}

# Test: start from handcrafted, replace ONE feature at a time with actual value
print("=== Single-feature ablation: handcrafted -> actual ===")
print(f"{'Feature':30s} {'Handcrafted':>12s} {'Actual':>12s} {'P(prof) with actual':>20s}")
print("-" * 80)
for feat in FEATURE_NAMES:
    if handcrafted[feat] != actual[feat]:
        test = handcrafted.copy()
        test[feat] = actual[feat]
        X = np.array([[test[f] for f in FEATURE_NAMES]])
        p = model.predict_proba(X)[0][1]
        print(f"  {feat:28s} {handcrafted[feat]:12.3f} {actual[feat]:12.3f} {p:20.4f}  {'<-- FLIPS' if p < 0.5 else ''}")

# Now test: substitute ALL differing features except the suspect
print()
print("=== Reverse ablation: actual values, restore ONE feature to handcrafted ===")
X_actual = np.array([[actual[f] for f in FEATURE_NAMES]])
p_actual = model.predict_proba(X_actual)[0][1]
print(f"  All actual: P(prof) = {p_actual:.4f}")
print()
for feat in FEATURE_NAMES:
    if handcrafted[feat] != actual[feat]:
        test = actual.copy()
        test[feat] = handcrafted[feat]
        X = np.array([[test[f] for f in FEATURE_NAMES]])
        p = model.predict_proba(X)[0][1]
        delta = p - p_actual
        print(f"  Restore {feat:28s} -> {handcrafted[feat]:8.3f}:  P(prof) = {p:.4f}  (delta {delta:+.4f})")
