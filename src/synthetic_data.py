"""
src/synthetic_data.py — §3.3

Generate labeled training data grounded in GPU performance physics.
This is what makes the model's rationale defensible instead of arbitrary.

External deps: numpy, pandas
Internal deps: feature_extractor.FEATURE_NAMES (import only, no parsing needed)

IMPORTANT (per spec): never report accuracy on this synthetic data as a
headline result — it's a sanity check only. The number that matters is
accuracy against real T4 benchmark measurements. Reporting synthetic-vs-
synthetic accuracy is circular, since the rule-based baseline is a
simplified version of the same heuristic used to generate these labels.
train.py prints this caveat every time it runs.
"""

from math import log2
import numpy as np
import pandas as pd

from src.feature_extractor import FEATURE_NAMES

# GPU/CPU physical constants used by the profitability formula.
GPU_PEAK_GFLOPS, CPU_PEAK_GFLOPS = 8100, 100   # T4 FP32 vs ~4GHz 8-wide AVX x4 core
PCIE_BW_GBPS, LAUNCH_OVERHEAD_US = 16, 10      # Updated PCIe 3.0 x16 to 16 GB/s

# 1.2x speedup threshold accounts for measurement noise.
PROFITABLE_THRESHOLD_LOG2 = log2(1.2)  # ≈ 0.263


def _generate_row(rng: np.random.Generator) -> dict:
    """Sample one correlated (not independent) set of loop parameters and
    label it via a physics-informed speedup estimate."""

    is_profitable_profile = rng.random() < 0.25
    if is_profitable_profile:
        trip_count_log = float(np.clip(rng.normal(loc=28, scale=3), 20, 32))
        nest_depth = int(rng.choice([1, 2, 3], p=[0.4, 0.4, 0.2]))
        n_arrays = int(np.clip(rng.poisson(lam=2) + 1, 1, 5))
        ops_per_iter = float(np.clip(rng.lognormal(mean=3.0, sigma=1.0), 10, 500))
        data_reuse = float(np.clip(rng.normal(0.3, 0.1), 0.1, 0.5))
        stride_one_ratio = float(np.clip(rng.normal(0.9, 0.1), 0.5, 1.0))
        has_indirect = 0
        has_dep = 0
        branch_count = 0
    else:
        trip_count_log = float(np.clip(rng.normal(loc=12, scale=4), 2, 25))
        nest_depth = int(rng.choice([0, 1, 2, 3], p=[0.3, 0.4, 0.2, 0.1]))
        n_arrays = int(np.clip(rng.poisson(lam=3) + 1, 1, 15))
        ops_per_iter = float(np.clip(rng.lognormal(mean=1.5, sigma=1.2), 1, 200))
        data_reuse = float(np.clip(rng.normal(0.2, 0.3), -1, 0.5))
        stride_one_ratio = float(rng.beta(3, 1.5))
        has_indirect = int(rng.binomial(1, 0.08))
        has_dep = int(rng.binomial(1, 0.10))
        branch_count = int(rng.poisson(1.5))

    inner_loop_count = max(0, nest_depth - 1) + rng.poisson(0.3)
    has_float_ops = int(rng.binomial(1, 0.6))
    mem_reads = int(n_arrays * rng.uniform(1, 3))
    mem_writes = int(n_arrays * rng.uniform(0.3, 1.0))
    unique_arrays = n_arrays
    comp_intensity = ops_per_iter / (mem_reads + mem_writes + 1)
    
    has_reduction = int(rng.binomial(1, 0.25))

    trip_count = 2 ** trip_count_log
    
    # Fix the O(N^3) memory footprint bug!
    # If a loop has nesting and temporal reuse, the true memory footprint is vastly smaller than trip_count.
    # We estimate the 1D dimension N, and reduce the effective trip count by N^reuse_score.
    effective_trip_count = trip_count
    if nest_depth > 0 and data_reuse > 0:
        n_approx = trip_count ** (1.0 / (nest_depth + 1))
        effective_trip_count = trip_count / (n_approx ** min(1.0, data_reuse))
        
    est_data_bytes = unique_arrays * max(1, effective_trip_count) * (
        8 if (has_float_ops and rng.random() > 0.5) else 4
    )
    
    estimated_data_bytes_log = log2(max(est_data_bytes, 1))
    total_ops_log = log2(max(ops_per_iter * trip_count, 1))
    transfer_penalty = estimated_data_bytes_log - total_ops_log

    # --- physics-informed speedup estimate ---
    gpu_compute_s = (ops_per_iter * trip_count) / (GPU_PEAK_GFLOPS * 1e9)
    transfer_s = est_data_bytes / (PCIE_BW_GBPS * 1e9)
    launch_s = LAUNCH_OVERHEAD_US * 1e-6
    gpu_total_s = gpu_compute_s + transfer_s + launch_s
    cpu_compute_s = (ops_per_iter * trip_count) / (CPU_PEAK_GFLOPS * 1e9)

    if has_dep:
        gpu_total_s *= 100          # dependency kills parallelism
    if has_indirect:
        gpu_total_s *= 3            # random access penalty
    if branch_count > 0:
        gpu_total_s *= 1 + 0.15 * branch_count   # ~15%/branch divergence
    gpu_total_s *= 1 + 2 * (1 - stride_one_ratio)  # non-coalesced penalty
    if data_reuse > 0.5:
        gpu_total_s *= 0.8          # shared-memory reuse benefit

    speedup = cpu_compute_s / max(gpu_total_s, 1e-15)
    speedup_log2 = log2(max(speedup, 0.01)) + rng.normal(0, 0.3)  # add noise for realism

    return {
        'trip_count_log': trip_count_log,
        'nest_depth': float(nest_depth),
        'inner_loop_count': float(inner_loop_count),
        'total_ops': ops_per_iter,
        'has_float_ops': float(has_float_ops),
        'comp_intensity': comp_intensity,
        'mem_reads': float(mem_reads),
        'mem_writes': float(mem_writes),
        'unique_arrays': float(unique_arrays),
        'stride_one_ratio': stride_one_ratio,
        'has_indirect_access': float(has_indirect),
        'data_reuse_score': data_reuse,
        'estimated_data_bytes_log': estimated_data_bytes_log,
        'transfer_penalty': transfer_penalty,
        'branch_count': float(branch_count),
        'has_reduction': float(has_reduction),
        'has_loop_carried_dep': float(has_dep),
        'speedup_log2': speedup_log2,
    }


def generate_dataset(n_samples: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Generate n_samples synthetic (features, speedup_log2) rows."""
    rng = np.random.default_rng(seed)
    rows = [_generate_row(rng) for _ in range(n_samples)]
    df = pd.DataFrame(rows, columns=FEATURE_NAMES + ['speedup_log2'])
    df['profitable'] = (df['speedup_log2'] > PROFITABLE_THRESHOLD_LOG2).astype(int)
    return df


_HANDCRAFTED_BENCHMARKS = [
    dict(name='GEMM_N2048', trip_count_log=33.0, nest_depth=2, inner_loop_count=1,
         total_ops=4.0, has_float_ops=1, comp_intensity=1.0, mem_reads=2, mem_writes=1,
         unique_arrays=3, stride_one_ratio=0.33, has_indirect_access=0, data_reuse_score=0.44,
         estimated_data_bytes_log=33.9, transfer_penalty=-1.0, branch_count=0,
         has_reduction=1, has_loop_carried_dep=0, speedup_log2=3.5),
    dict(name='GEMM_N32', trip_count_log=15.0, nest_depth=2, inner_loop_count=1,
         total_ops=4.0, has_float_ops=1, comp_intensity=1.0, mem_reads=2, mem_writes=1,
         unique_arrays=3, stride_one_ratio=0.33, has_indirect_access=0, data_reuse_score=0.44,
         estimated_data_bytes_log=18.0, transfer_penalty=1.0, branch_count=0,
         has_reduction=1, has_loop_carried_dep=0, speedup_log2=-1.0),
    dict(name='VectorAdd_1M', trip_count_log=20.0, nest_depth=0, inner_loop_count=0,
         total_ops=1.0, has_float_ops=1, comp_intensity=0.33, mem_reads=2, mem_writes=1,
         unique_arrays=3, stride_one_ratio=1.0, has_indirect_access=0, data_reuse_score=0.0,
         estimated_data_bytes_log=24.5, transfer_penalty=4.5, branch_count=0,
         has_reduction=0, has_loop_carried_dep=0, speedup_log2=-0.4),
    dict(name='Jacobi2D_1024', trip_count_log=20.0, nest_depth=1, inner_loop_count=1,
         total_ops=9.0, has_float_ops=1, comp_intensity=1.5, mem_reads=4, mem_writes=1,
         unique_arrays=2, stride_one_ratio=0.5, has_indirect_access=0, data_reuse_score=0.2,
         estimated_data_bytes_log=24.0, transfer_penalty=0.8, branch_count=0,
         has_reduction=0, has_loop_carried_dep=0, speedup_log2=2.4),
    dict(name='Histogram_scatter', trip_count_log=20.0, nest_depth=0, inner_loop_count=0,
         total_ops=1.0, has_float_ops=0, comp_intensity=0.5, mem_reads=2, mem_writes=1,
         unique_arrays=2, stride_one_ratio=0.1, has_indirect_access=1, data_reuse_score=0.0,
         estimated_data_bytes_log=23.5, transfer_penalty=3.5, branch_count=0,
         has_reduction=0, has_loop_carried_dep=0, speedup_log2=-2.0),
    dict(name='PrefixScan_naive', trip_count_log=20.0, nest_depth=0, inner_loop_count=0,
         total_ops=1.0, has_float_ops=1, comp_intensity=1.0, mem_reads=2, mem_writes=1,
         unique_arrays=2, stride_one_ratio=1.0, has_indirect_access=0, data_reuse_score=0.0,
         estimated_data_bytes_log=23.0, transfer_penalty=3.0, branch_count=0,
         has_reduction=0, has_loop_carried_dep=1, speedup_log2=-3.0),
    dict(name='SYRK_N1024', trip_count_log=30.0, nest_depth=2, inner_loop_count=1,
         total_ops=2.0, has_float_ops=1, comp_intensity=0.66, mem_reads=2, mem_writes=1,
         unique_arrays=2, stride_one_ratio=0.33, has_indirect_access=0, data_reuse_score=0.44,
         estimated_data_bytes_log=30.0, transfer_penalty=-1.0, branch_count=0,
         has_reduction=1, has_loop_carried_dep=0, speedup_log2=3.1),
    dict(name='BranchHeavyFilter', trip_count_log=20.0, nest_depth=0, inner_loop_count=0,
         total_ops=3.0, has_float_ops=0, comp_intensity=0.7, mem_reads=2, mem_writes=1,
         unique_arrays=2, stride_one_ratio=0.95, has_indirect_access=0, data_reuse_score=0.0,
         estimated_data_bytes_log=23.0, transfer_penalty=1.5, branch_count=8,
         has_reduction=0, has_loop_carried_dep=0, speedup_log2=-0.8),
    dict(name='PointerChase_LinkedList', trip_count_log=16.0, nest_depth=0, inner_loop_count=0,
         total_ops=2.0, has_float_ops=0, comp_intensity=0.5, mem_reads=1, mem_writes=0,
         unique_arrays=1, stride_one_ratio=0.0, has_indirect_access=1, data_reuse_score=0.0,
         estimated_data_bytes_log=18.0, transfer_penalty=1.0, branch_count=0,
         has_reduction=0, has_loop_carried_dep=1, speedup_log2=-4.0),
    dict(name='TinyLoop_16cubed', trip_count_log=12.0, nest_depth=2, inner_loop_count=1,
         total_ops=1.0, has_float_ops=1, comp_intensity=1.0, mem_reads=2, mem_writes=1,
         unique_arrays=2, stride_one_ratio=1.0, has_indirect_access=0, data_reuse_score=0.0,
         estimated_data_bytes_log=15.0, transfer_penalty=3.0, branch_count=0,
         has_reduction=0, has_loop_carried_dep=0, speedup_log2=-3.5),
    dict(name='Conv2D_3x3_N1024', trip_count_log=22.0, nest_depth=3, inner_loop_count=3,
         total_ops=9.0, has_float_ops=1, comp_intensity=2.25, mem_reads=2, mem_writes=1,
         unique_arrays=3, stride_one_ratio=0.16, has_indirect_access=0, data_reuse_score=0.33,
         estimated_data_bytes_log=24.7, transfer_penalty=-0.4, branch_count=0,
         has_reduction=1, has_loop_carried_dep=0, speedup_log2=4.5),
    dict(name='NBody_N16384', trip_count_log=28.0, nest_depth=1, inner_loop_count=1,
         total_ops=19.0, has_float_ops=1, comp_intensity=1.9, mem_reads=6, mem_writes=3,
         unique_arrays=6, stride_one_ratio=0.66, has_indirect_access=0, data_reuse_score=0.33,
         estimated_data_bytes_log=28.9, transfer_penalty=-3.3, branch_count=0,
         has_reduction=1, has_loop_carried_dep=0, speedup_log2=5.0),
    dict(name='LBM_FluidSim', trip_count_log=20.0, nest_depth=1, inner_loop_count=1,
         total_ops=17.0, has_float_ops=1, comp_intensity=1.88, mem_reads=6, mem_writes=2,
         unique_arrays=4, stride_one_ratio=0.5, has_indirect_access=0, data_reuse_score=0.125,
         estimated_data_bytes_log=23.7, transfer_penalty=-0.33, branch_count=0,
         has_reduction=1, has_loop_carried_dep=0, speedup_log2=3.8),
]


def add_benchmark_examples(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for b in _HANDCRAFTED_BENCHMARKS:
        row = {k: float(v) for k, v in b.items() if k != 'name' and k != 'speedup_log2'}
        row['speedup_log2'] = float(b['speedup_log2'])
        rows.append(row)
    bench_df = pd.DataFrame(rows, columns=FEATURE_NAMES + ['speedup_log2'])
    bench_df['profitable'] = (bench_df['speedup_log2'] > PROFITABLE_THRESHOLD_LOG2).astype(int)
    return pd.concat([df, bench_df], ignore_index=True)


if __name__ == '__main__':
    data = add_benchmark_examples(generate_dataset())
    print(data.shape)
    print(data['profitable'].value_counts())
    print(data.head())