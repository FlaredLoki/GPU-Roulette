"""
src/baseline.py — §3.5

Rule-based classifier — the "dumb baseline" the model has to actually beat.
This is your evidence that ML adds value, so the thresholds are calibrated
to the synthetic data's scale (e.g. trip_count_log > 17 means >131K
iterations).
"""

import numpy as np


class RuleBasedClassifier:
    def predict_single(self, features: dict) -> tuple:
        if features['has_loop_carried_dep'] > 0.5:
            return (0, 0.95, "Loop-carried dependency prevents parallelization")
        if features['trip_count_log'] < 7:
            return (0, 0.90, "Insufficient parallelism (< 128 iterations)")
        if features['has_indirect_access'] > 0.5 and features['comp_intensity'] < 1.0:
            return (0, 0.80, "Indirect memory access with low compute intensity")
        if (features['trip_count_log'] > 13 and features['comp_intensity'] > 2.0
                and features['stride_one_ratio'] > 0.5 and features['branch_count'] < 3):
            return (1, 0.90, "High parallelism, compute-bound, coalesced access")
        if features['trip_count_log'] > 17 and features['comp_intensity'] > 1.0:
            return (1, 0.85, "Very high parallelism with adequate compute intensity")
        if features['transfer_penalty'] > 5 and features['trip_count_log'] < 15:
            return (0, 0.75, "Data transfer cost dominates computation")
        if features['branch_count'] > 6:
            return (0, 0.70, "Excessive branch divergence")
        if features['trip_count_log'] > 10 and features['comp_intensity'] > 1.0:
            return (1, 0.60, "Moderate parallelism and compute intensity")
        return (0, 0.55, "Insufficient evidence for GPU profitability")

    def predict(self, X):
        """X: DataFrame or list of feature dicts. Loops predict_single per row."""
        rows = X.to_dict('records') if hasattr(X, 'to_dict') else X
        return np.array([self.predict_single(row)[0] for row in rows])

    def predict_proba(self, X):
        """Returns [1-conf, conf] per row (sklearn-compatible interface)."""
        rows = X.to_dict('records') if hasattr(X, 'to_dict') else X
        out = []
        for row in rows:
            verdict, conf, _ = self.predict_single(row)
            out.append([1 - conf, conf] if verdict == 1 else [conf, 1 - conf])
        return np.array(out)