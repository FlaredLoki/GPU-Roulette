"""
Fix suggestions for unprofitable GPU offload candidates.

Given extracted features and dependency info, generates actionable
code transformation suggestions — each tied to a specific GPU hardware
mechanism. Rules are ordered by impact: memory coalescing first (most
common, biggest payoff), then trip count, dependency, divergence,
indirect access, and transfer penalty.

External deps: None
Internal deps: None (uses raw feature dict and DependencyInfo)
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.feature_extractor import DependencyInfo


def suggest_fixes(
    features: dict[str, float],
    dependency: 'DependencyInfo',
) -> list[dict]:
    """
    Generate code transformation suggestions for an unprofitable loop.

    Each suggestion is a dict with:
      problem:              What's wrong (human-readable)
      fix:                  What to do about it
      expected_improvement: Rough estimate of benefit
      code_example:         Before/after snippet

    Args:
        features: 17-feature dict from extract_features().
        dependency: DependencyInfo from extract_features().

    Returns:
        List of fix dicts, ordered by likely impact.
    """
    fixes = []

    # 1. Non-coalesced access
    if features['stride_one_ratio'] < 0.5:
        fixes.append({
            "problem": (
                f"Only {features['stride_one_ratio']*100:.0f}% of "
                "memory accesses are coalesced"
            ),
            "fix": (
                "Apply loop interchange — swap inner and outer loops "
                "so the fastest-varying index matches array layout"
            ),
            "expected_improvement": "2-10x memory bandwidth improvement",
            "code_example": (
                "Before: for(i) for(j) A[j][i]\n"
                "After:  for(j) for(i) A[j][i]"
            ),
        })

    # 2. Low trip count
    if features['trip_count_log'] < 10:  # < 1024
        fixes.append({
            "problem": (
                f"Low trip count (~{2**features['trip_count_log']:.0f} "
                "iterations)"
            ),
            "fix": (
                "Consider loop fusion with adjacent loop, or guard "
                "offload with runtime size check"
            ),
            "expected_improvement": (
                "Amortize kernel launch overhead across more work"
            ),
            "code_example": "if (N > 4096) { #pragma omp target ... }",
        })

    # 3. Loop-carried dependency
    if dependency.category == "DEPENDENCY":
        fixes.append({
            "problem": (
                "Loop-carried dependency prevents parallelization"
            ),
            "fix": (
                "Consider algorithmic change: parallel prefix sum "
                "(Blelloch scan) for accumulations, or split into "
                "independent phases"
            ),
            "expected_improvement": (
                "Enables parallelization if algorithm permits"
            ),
            "code_example": (
                "// Sequential: a[i] = a[i-1] + x[i]\n"
                "// Parallel: use two-phase up-sweep/down-sweep"
            ),
        })

    # 4. High branch count
    if features['branch_count'] > 4:
        fixes.append({
            "problem": (
                f"{int(features['branch_count'])} branches cause "
                "warp divergence"
            ),
            "fix": (
                "Replace branches with branchless arithmetic: "
                "max(x,0) instead of if(x>0)"
            ),
            "expected_improvement": (
                "Reduce warp serialization, improve SIMT utilization"
            ),
            "code_example": (
                "// Before: if(x > 0) y = x; else y = 0;\n"
                "// After:  y = fmax(x, 0.0f);"
            ),
        })

    # 5. Indirect access
    if features['has_indirect_access'] > 0.5:
        fixes.append({
            "problem": (
                "Indirect array access (A[B[i]]) causes random "
                "memory access"
            ),
            "fix": (
                "Sort data by access pattern before offloading, "
                "or restructure to direct indexing"
            ),
            "expected_improvement": (
                "Improve memory coalescing and cache hit rate"
            ),
            "code_example": (
                "// Pre-sort indices so access pattern becomes sequential"
            ),
        })

    # 6. High transfer penalty
    if features['transfer_penalty'] > 3:
        fixes.append({
            "problem": "Data transfer cost dominates computation",
            "fix": (
                "Wrap multiple kernels in #pragma omp target data to "
                "keep data resident on GPU across calls"
            ),
            "expected_improvement": (
                "Eliminate redundant host\u2194device transfers"
            ),
            "code_example": (
                "#pragma omp target data map(tofrom: A[0:N])\n"
                "{\n"
                "  // multiple kernels here \u2014 data stays on GPU\n"
                "}"
            ),
        })

    return fixes
