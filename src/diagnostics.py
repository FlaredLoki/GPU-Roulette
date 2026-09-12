"""
src/diagnostics.py — §3.12

Purpose: format predictions like a real compiler diagnostic (clang -Rpass
style) — this is what makes the CLI output look like a credible tool
instead of a script dump.
"""


def format_diagnostic(result, filename: str = "<stdin>") -> str:
    """
    {filename}:{line}:{col}: remark: GPU offload profitability analysis [-Rgpu-gate]
    {filename}:{line}:{col}: remark: {VERDICT} — {confidence}% confidence
    {filename}:{line}:{col}: note: {top_positive_factor}
    {filename}:{line}:{col}: note: {second_positive_factor}
    {filename}:{line}:{col}: warning: {top_negative_factor}
    {filename}:{line}:{col}: note: breakeven problem size N >= {crossover_n}
    {filename}:{line}:{col}: note: suggested directive:
    {filename}:{line}:{col}: note:   #pragma omp target teams distribute parallel for ...

    Top 3 positive SHAP contributors as note:, top 2 negative as warning:,
    blockers as error:, fix suggestions as note: if unprofitable.
    """
    line = getattr(result, 'loop_line', 0)
    col = 1
    loc = f"{filename}:{line}:{col}"

    ranked = sorted(result.shap_values.items(), key=lambda kv: kv[1], reverse=True)
    positives = [(n, v) for n, v in ranked if v > 0][:3]
    negatives = sorted(
        [(n, v) for n, v in ranked if v <= 0], key=lambda kv: kv[1]
    )[:2]

    out = [
        f"{loc}: remark: GPU offload profitability analysis [-Rgpu-gate]",
        f"{loc}: remark: {result.verdict} \u2014 {result.confidence*100:.0f}% confidence",
    ]

    for name, val in positives:
        out.append(f"{loc}: note: {name} contributes toward offload [{val:+.2f}]")

    for name, val in negatives:
        out.append(f"{loc}: warning: {name} works against offload [{val:+.2f}]")

    for blocker in (result.blockers or []):
        msg = blocker.get('reason', str(blocker)) if isinstance(blocker, dict) else str(blocker)
        out.append(f"{loc}: error: {msg}")

    if result.crossover_n is not None:
        out.append(f"{loc}: note: breakeven problem size N >= {result.crossover_n}")

    if result.verdict != "PROFITABLE":
        for fix in (result.fix_suggestions or []):
            msg = (fix.get('problem', '') + ' -> ' + fix.get('fix', '')) if isinstance(fix, dict) else str(fix)
            out.append(f"{loc}: note: {msg}")

    out.append(f"{loc}: note: suggested directive:")
    directive = _suggest_directive(result)
    out.append(f"{loc}: note:   {directive}")

    return "\n".join(out)


def _suggest_directive(result) -> str:
    """Base clause + reduction clauses for detected accumulators,
    collapse(nest_depth+1) if nested, map(tofrom: ...) for detected arrays."""
    base = "#pragma omp target teams distribute parallel for"
    clauses = []

    dep_category = None
    if hasattr(result, 'feature_vector') and result.feature_vector is not None:
        dep_category = getattr(result.feature_vector.dependency, 'category', None)
        nest_depth = result.feature_vector.features.get('nest_depth', 0)
        if dep_category == "REDUCTION":
            # Use detected accumulators if available, otherwise fall back
            accumulators = getattr(result.feature_vector.dependency, 'accumulators', None)
            if accumulators:
                for var, op in accumulators:
                    clauses.append(f"reduction({op}:{var})")
            else:
                clauses.append("reduction(+:sum)")
        if nest_depth and nest_depth > 0:
            clauses.append(f"collapse({int(nest_depth) + 1})")
        clauses.append("map(tofrom: ...)")

    if clauses:
        return base + " " + " ".join(clauses)
    return base + " ..."


def format_batch_summary(results: list, filename: str) -> str:
    """Ranked table across multiple loop regions, PROFITABLE first (by
    confidence desc), then UNPROFITABLE, then BLOCKED."""
    order = {"PROFITABLE": 0, "UNPROFITABLE": 1, "UNCERTAIN": 2, "BLOCKED": 3}
    ranked = sorted(
        results,
        key=lambda r: (order.get(r.verdict, 99), -r.confidence)
    )

    header = f"{'Line':>6}  {'Verdict':<12}  {'Confidence':>10}  Notes"
    sep = "-" * len(header)
    lines = [f"=== GPUGate summary: {filename} ===", header, sep]

    for r in ranked:
        note = ""
        if r.verdict == "BLOCKED" and r.blockers:
            first = r.blockers[0]
            note = first.get('reason', str(first)) if isinstance(first, dict) else str(first)
        elif r.crossover_n is not None:
            note = f"breakeven N >= {r.crossover_n}"
        lines.append(f"{r.loop_line:>6}  {r.verdict:<12}  {r.confidence*100:>9.0f}%  {note}")

    return "\n".join(lines)