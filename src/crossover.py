"""
src/crossover.py — §3.9

Purpose: find the breakeven problem size N* where offload becomes
profitable — turns a binary verdict into an actionable "at what size does
this pay off."
"""

from math import log2
from typing import Optional


def find_breakeven(model, features: dict, nest_depth: int, n_arrays: int,
                    element_size: int) -> Optional[int]:
    """No defaults on the last three params — always pass the real values
    from feature_vector.

    Sweeps N from 16 to 2**20 (~50 log-spaced points). Per N, with
    total_trip = N ** (nest_depth + 1):
        features['trip_count_log'] = log2(total_trip)
        features['estimated_data_bytes_log'] = log2(total_trip * n_arrays * element_size)
        features['transfer_penalty'] = estimated_data_bytes_log - log2(total_ops_per_iter * total_trip)
    then get model.predict_proba(...). See _sweep_probabilities().

    Find where probability crosses 0.5 (linear interpolation), round to
    nearest power of 2. None if it never crosses.
    """
    features = features.copy()  # sweep mutates the dict; don't corrupt caller's FeatureVector
    ns, probs = _sweep_probabilities(model, features, nest_depth, n_arrays, element_size)

    for i in range(len(ns) - 1):
        p0, p1 = probs[i], probs[i + 1]
        if p0 < 0.5 <= p1:
            # linear interpolation between ns[i] and ns[i+1] in log-space
            if p1 == p0:
                n_cross = ns[i]
            else:
                frac = (0.5 - p0) / (p1 - p0)
                log_n = _log2(ns[i]) + frac * (_log2(ns[i + 1]) - _log2(ns[i]))
                n_cross = 2 ** log_n
            return _round_to_power_of_2(n_cross)

    return None


def generate_profitability_curve(model, features: dict, nest_depth: int,
                                  n_arrays: int, element_size: int) -> list:
    """Same sweep, returns the full [{"N": ..., "probability": ...}, ...]
    list for plotting (used by the web UI if you get to it)."""
    features = features.copy()
    ns, probs = _sweep_probabilities(model, features, nest_depth, n_arrays, element_size)
    return [{"N": n, "probability": p} for n, p in zip(ns, probs)]


# --- helpers ----------------------------------------------------------------

def _sweep_probabilities(model, features: dict, nest_depth: int, n_arrays: int,
                          element_size: int) -> tuple:
    """Shared sweep logic for find_breakeven() and generate_profitability_curve().

    total_trip = N ** (nest_depth + 1): the candidate loop itself always
    contributes one factor of N, plus nest_depth more nested levels
    (e.g. GEMM with nest_depth=2 -> N**3, not N**2). Used uniformly for
    both trip_count_log and the data-bytes line.

    transfer_penalty is recomputed at each swept N as
    est_bytes_log - log2(total_ops_per_iter * total_trip) - the same
    unit (log2 of a byte-scale quantity minus log2 of an op-scale
    quantity) synthetic_data.py uses, instead of subtracting a raw
    small op count directly from a log-byte value.
    """
    total_ops_per_iter = features.get('total_ops', 1.0)
    data_reuse = features.get('data_reuse_score', 0.0)

    ns = _log_spaced_ints(16, 2 ** 20, num=50)
    probs = []
    for n in ns:
        total_trip = n ** (nest_depth + 1)
        trip_count_log = log2(max(total_trip, 1))
        
        effective_trip_count = total_trip
        if nest_depth > 0 and data_reuse > 0:
            n_approx = total_trip ** (1.0 / (nest_depth + 1))
            effective_trip_count = total_trip / (n_approx ** min(1.0, data_reuse))
            
        est_bytes_log = log2(max(effective_trip_count * n_arrays * element_size, 1))
        ops_total_log = log2(max(total_ops_per_iter * total_trip, 1))

        features['trip_count_log'] = trip_count_log
        features['estimated_data_bytes_log'] = est_bytes_log
        features['transfer_penalty'] = est_bytes_log - ops_total_log

        probs.append(_predict_positive_proba(model, features))

    return ns, probs


def _predict_positive_proba(model, features: dict) -> float:
    from src.feature_extractor import FEATURE_NAMES
    import numpy as np
    arr = np.array([[features[name] for name in FEATURE_NAMES]], dtype=float)
    return float(model.predict_proba(arr)[0][1])


def _log2(x: float) -> float:
    return log2(x)


def _log_spaced_ints(lo: int, hi: int, num: int) -> list:
    import numpy as np
    vals = np.logspace(log2(lo), log2(hi), num=num, base=2)
    ints = sorted(set(int(round(v)) for v in vals))
    return ints


def _round_to_power_of_2(n: float) -> int:
    import math
    return int(2 ** round(math.log2(max(n, 1))))