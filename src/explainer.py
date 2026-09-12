"""
src/explainer.py — §3.8

Purpose: turn SHAP values + features into the natural-language explanation
that's the actual differentiator of this project.

Internal deps: feature_extractor.FEATURE_GPU_CONCEPTS, feature_extractor.FEATURE_GROUPS
(placeholders here — swap for your teammate's tuned versions, see
feature_extractor.py header).
"""

from src.feature_extractor import FEATURE_GPU_CONCEPTS, FEATURE_GROUPS

TOP_K_CONTRIBUTORS = 5

# ---------------------------------------------------------------------------
# PLACEHOLDER: per-feature value -> human-readable description.
# The spec says the full, tuned lambda dict for all 17 features lives in
# main spec §3.8 ("copy it directly, it's tuned wording") — swap this in
# when you have it. These are functional stand-ins, not the tuned copy.
# ---------------------------------------------------------------------------

FEATURE_VALUE_DESCRIPTIONS = {
    'trip_count_log':           lambda v: f"~{2**v:,.0f} iterations",
    'nest_depth':                lambda v: f"{int(round(v))} nested loop level(s)",
    'inner_loop_count':          lambda v: f"{int(round(v))} inner loop(s)",
    'total_ops':                 lambda v: f"{v:.0f} ops/iteration",
    'has_float_ops':             lambda v: "uses floating-point ops" if v > 0.5 else "integer-only ops",
    'comp_intensity':            lambda v: f"{v:.2f} ops/access",
    'mem_reads':                 lambda v: f"{v:.0f} memory reads/iteration",
    'mem_writes':                lambda v: f"{v:.0f} memory writes/iteration",
    'unique_arrays':             lambda v: f"{v:.0f} distinct array(s)",
    'stride_one_ratio':          lambda v: f"{v*100:.0f}% stride-1 (coalesced)",
    'has_indirect_access':       lambda v: "indirect/gather-scatter access" if v > 0.5 else "direct access",
    'data_reuse_score':          lambda v: f"{v:+.2f} data reuse score",
    'estimated_data_bytes_log':  lambda v: f"~{2**v/1e6:,.1f} MB transferred",
    'transfer_penalty':          lambda v: f"{v:+.1f} log2(bytes/op) transfer-to-compute ratio",
    'branch_count':              lambda v: f"{v:.0f} branch(es) in loop body",
    'has_reduction':             lambda v: "reduction pattern present" if v > 0.5 else "no reduction pattern",
    'has_loop_carried_dep':      lambda v: "loop-carried dependency present" if v > 0.5 else "no loop-carried dependency",
}


def feature_value_description(feature_name: str, value: float) -> str:
    fn = FEATURE_VALUE_DESCRIPTIONS.get(feature_name)
    if fn is None:
        return f"{value:.3f}"
    return fn(value)


def group_shap_values(shap_values: dict) -> dict:
    """Sum SHAP values per FEATURE_GROUPS entry."""
    grouped = {}
    for feature, value in shap_values.items():
        group = FEATURE_GROUPS.get(feature, "Other")
        grouped[group] = grouped.get(group, 0.0) + value
    return grouped


def generate_explanation(verdict: str, confidence: float, features: dict,
                          shap_values: dict, dependency, confidence_limited: bool,
                          confidence_reason: str) -> str:
    """Sort features by |SHAP value| descending, take top 5. Split positive
    (-> profitable) vs negative (-> unprofitable) contributors."""
    ranked = sorted(shap_values.items(), key=lambda kv: abs(kv[1]), reverse=True)
    top = ranked[:TOP_K_CONTRIBUTORS]

    positives = [(name, val) for name, val in top if val > 0]
    negatives = [(name, val) for name, val in top if val <= 0]

    lines = [f"GPU-{verdict.upper()} ({confidence*100:.0f}% confidence)", ""]

    if positives:
        lines.append("Primary factors:")
        for name, val in positives:
            concept = FEATURE_GPU_CONCEPTS.get(name, name)
            desc = feature_value_description(name, features[name])
            lines.append(f"\u2713 {concept}: {desc} [{val:+.2f}]")
        lines.append("")

    if negatives:
        lines.append("Minor concerns:" if positives else "Primary concerns:")
        for name, val in negatives:
            concept = FEATURE_GPU_CONCEPTS.get(name, name)
            desc = feature_value_description(name, features[name])
            lines.append(f"\u26a0 {concept}: {desc} [{val:+.2f}]")
        lines.append("")

    if dependency is not None and getattr(dependency, 'category', 'NONE') != 'NONE':
        lines.append(f"Dependency note ({dependency.category}): {dependency.details}")
        lines.append("")

    if confidence_limited:
        lines.append(f"\u26a0 Confidence limited: {confidence_reason}")
        lines.append("")

    return "\n".join(lines).rstrip()


def generate_grouped_explanation(grouped_shap: dict, features: dict) -> str:
    """5-group summary (Parallelism Potential / Compute Characteristics /
    Memory Efficiency / Data Transfer Cost / Control Flow Overhead), each a
    summed SHAP value, plus a net total and final verdict line."""
    order = [
        'Parallelism Potential', 'Compute Characteristics', 'Memory Efficiency',
        'Data Transfer Cost', 'Control Flow Overhead',
    ]
    lines = ["Group summary:"]
    net_total = 0.0
    for group in order:
        val = grouped_shap.get(group, 0.0)
        net_total += val
        sign = "+" if val >= 0 else ""
        lines.append(f"  {group:26s}: {sign}{val:.3f}")

    lines.append(f"  {'Net total':26s}: {'+' if net_total >= 0 else ''}{net_total:.3f}")
    lines.append("")
    lines.append(
        "Final verdict: GPU-PROFITABLE" if net_total > 0 else "Final verdict: GPU-UNPROFITABLE"
    )
    return "\n".join(lines)