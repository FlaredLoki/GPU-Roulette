# Known Limitations

This document outlines the known boundaries and limitations of the current static extraction and modeling pipeline. These design choices or current missing capabilities were intentionally deferred during development to focus on end-to-end functionality.

### Sibling inner loops undercounted
The total loop trip-count estimation (`_compute_total_trip_count_log`) uses `max()` instead of `sum()` when evaluating sequential (non-nested) loops that share the same parent loop body. While this correctly preserves the algebraic complexity class (e.g., $O(N^2)$) and avoids wildly overestimating total work, it causes the literal iteration count for back-to-back sibling loops to be undercounted.

### `while` and `do-while` induction variables never detected
The parsing logic currently hardcodes `induction_var=None` when processing `while` and `do-while` constructs (`_parse_while_loop` and `_parse_do_while_loop`). Because the induction variable is missing, downstream memory feature extraction (such as `stride_one_ratio` and `data_reuse_score`) cannot determine which index expressions are tied to the loop, causing these memory features to silently fall back to 0.0 for all `while` and `do-while` loops.

### Loop step size, direction, and early exits ignored in trip-count estimation
The trip-count estimator (`_estimate_trip_count`) only evaluates the literal bound found in the loop condition (e.g., `N` in `i < N`) and completely ignores the loop's update expression. As a result, loops with custom strides (e.g., `i += 4`) or descending directions (e.g., `i--`) produce incorrect total trip counts. Furthermore, data-dependent early exits like `break`, `continue`, `goto`, and `return` within the loop body are invisible to the static iteration count. While the pipeline now recognizes these keywords and aggressively caps prediction confidence to 65% when they are present, it still calculates the raw memory and compute footprint as if the loop always runs to completion.

### Zero-iteration (dead) loops misreported as ~1000 iterations
If a loop's condition statically guarantees zero iterations (e.g., `for (int i = 0; i < 0; i++)`), the pipeline treats the evaluated bound (`0`) the exact same way it treats an unknown or symbolic variable bound. It falls into a generic fallback path and assumes a default iteration count of 1,000, rather than correctly identifying the loop as dead code.

### No true pointer-alias analysis
Dependency detection relies on purely syntactic AST pattern matching. While the `UNCERTAIN` dependency category exists and correctly throttles model confidence, it currently only fires for raw pointer-dereference patterns (e.g., `*p++ = ...`). If two differently-named pointer parameters are passed without the `restrict` keyword and accessed via array subscripts, the pipeline assumes they do not alias and will confidently classify the loop (usually as `NONE`), despite the potential for hidden memory overlaps in C.

### Malformed/syntactically broken C can still produce a full analysis
Because Tree-sitter is highly error-tolerant, it will often build a partial AST for syntactically broken code (e.g., missing semicolons or unclosed braces). This allows malformed loops to run completely through the extraction and prediction pipeline, occasionally resulting in confident-looking profitability verdicts for invalid code. This is partially mitigated by a warning banner surfaced in the UI indicating that the source contains syntax errors and the analysis may be incomplete.
