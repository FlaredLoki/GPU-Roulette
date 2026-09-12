"""
Roofline model positioning for GPU offload analysis.

Computes where a loop sits on the roofline diagram — the standard way
to visualize whether a workload is compute-bound or memory-bound on a
given GPU. The key metric is arithmetic intensity (FLOP/byte): below
the ridge point, the kernel is memory-bound; above it, compute-bound.

External deps: json, os
Internal deps: None (uses raw feature dict)
"""

import json
import os

# ---------------------------------------------------------------------------
# Default path to GPU specs data
# ---------------------------------------------------------------------------

_SPECS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'data', 'gpu_specs.json'
)


def get_gpu_specs(gpu_name: str = "T4") -> dict:
    """
    Load GPU hardware specifications from data/gpu_specs.json.

    Args:
        gpu_name: Key in the JSON file ("T4", "A100", "RTX3060").

    Returns:
        Dict with name, peak_gflops_fp32, mem_bandwidth_gbps,
        pcie_bandwidth_gbps, ridge_point_fp32, etc.

    Raises:
        FileNotFoundError: If gpu_specs.json is missing.
        KeyError: If gpu_name is not in the file.
    """
    with open(_SPECS_PATH, 'r') as f:
        all_specs = json.load(f)

    if gpu_name not in all_specs:
        available = ', '.join(all_specs.keys())
        raise KeyError(
            f"Unknown GPU '{gpu_name}'. Available: {available}"
        )

    return all_specs[gpu_name]


def compute_ridge_point(gpu_specs: dict) -> float:
    """
    Compute the roofline ridge point for a GPU.

    Ridge point = peak_gflops / mem_bandwidth_gbps (FLOP/byte).
    Below this arithmetic intensity: memory-bound.
    Above this arithmetic intensity: compute-bound.

    Args:
        gpu_specs: Dict from get_gpu_specs().

    Returns:
        Ridge point in FLOP/byte.
    """
    return gpu_specs['peak_gflops_fp32'] / gpu_specs['mem_bandwidth_gbps']


def compute_position(features: dict, element_size: int) -> dict:
    """
    Compute where a loop sits on the roofline diagram.

    Args:
        features: 17-feature dict from extract_features().
        element_size: Detected element size in bytes (1/2/4/8).
                      NO DEFAULT — caller must pass the real detected value.

    Returns:
        Dict with:
          arithmetic_intensity: FLOP/byte (x-axis of roofline)
          estimated_gflops: estimated total GFLOPS (y-axis)
          boundedness: 'compute-bound' or 'memory-bound' relative to T4
          ridge_point: the T4 ridge point used for classification

    Note:
        comp_intensity (ops/access, dimensionless) and arithmetic_intensity
        (FLOP/byte, physical) are deliberately different values — don't
        conflate them. The division by element_size converts from
        "operations per memory access" to "operations per byte."
    """
    # Arithmetic intensity in FLOP/byte (not ops/access!)
    ops_per_access = features['comp_intensity']
    arithmetic_intensity = ops_per_access / element_size

    # Estimated achievable performance
    trip_count = 2 ** features['trip_count_log']
    total_flops = features['total_ops'] * trip_count
    estimated_gflops = total_flops / 1e9

    # Classify against T4 ridge point
    try:
        specs = get_gpu_specs("T4")
        ridge_point = compute_ridge_point(specs)
    except (FileNotFoundError, KeyError):
        # Fallback: T4 ridge point ≈ 25.3
        ridge_point = 25.3

    return {
        'arithmetic_intensity': round(arithmetic_intensity, 4),
        'estimated_gflops': round(estimated_gflops, 6),
        'boundedness': 'compute-bound' if arithmetic_intensity > ridge_point else 'memory-bound',
        'ridge_point': round(ridge_point, 2),
    }
