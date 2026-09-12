"""
Static feature extractor for GPU offload profitability prediction.

Extracts 17 features from a parsed C loop that map directly to GPU
performance mechanisms. Each feature is designed to be interpretable
via SHAP explanations grounded in GPU hardware concepts.

Feature groups:
  Parallelism (3):  trip_count_log, nest_depth, inner_loop_count
  Compute    (3):  total_ops, has_float_ops, comp_intensity
  Memory     (6):  mem_reads, mem_writes, unique_arrays, stride_one_ratio,
                   has_indirect_access, data_reuse_score
  Transfer   (2):  estimated_data_bytes_log, transfer_penalty
  Control    (3):  branch_count, has_reduction, has_loop_carried_dep
"""

import math
from dataclasses import dataclass, field
from typing import Optional

from tree_sitter import Node

from src.code_parser import (
    LoopInfo,
    _node_text,
    count_node_type,
    collect_nodes,
    _find_first,
    find_offload_blockers,
)


# Feature names in canonical order — used by model and explainer
FEATURE_NAMES = [
    # Parallelism
    'trip_count_log',
    'nest_depth',
    'inner_loop_count',
    # Compute
    'total_ops',
    'has_float_ops',
    'comp_intensity',
    # Memory
    'mem_reads',
    'mem_writes',
    'unique_arrays',
    'stride_one_ratio',
    'has_indirect_access',
    'data_reuse_score',
    # Transfer
    'estimated_data_bytes_log',
    'transfer_penalty',
    # Control flow
    'branch_count',
    'has_reduction',
    'has_loop_carried_dep',
]

# Maps feature name → GPU hardware concept for explanations
FEATURE_GPU_CONCEPTS = {
    'trip_count_log': (
        'SM occupancy',
        'Need ~10K+ threads to saturate GPU streaming multiprocessors'
    ),
    'nest_depth': (
        'Parallelism dimensions',
        'Maps to CUDA grid/block structure — more dimensions = more threads'
    ),
    'inner_loop_count': (
        'Per-thread sequential work',
        'Inner sequential loops increase work per thread'
    ),
    'total_ops': (
        'Arithmetic throughput',
        'Raw compute work available to GPU ALUs per iteration'
    ),
    'has_float_ops': (
        'FP unit utilization',
        'GPU floating-point units are massively parallel; integer ops less so'
    ),
    'comp_intensity': (
        'Roofline position',
        'Ops per memory access — high = compute-bound (ideal for GPU)'
    ),
    'mem_reads': (
        'Memory bandwidth demand',
        'Array reads drive global memory traffic'
    ),
    'mem_writes': (
        'Write-back traffic',
        'Array writes add to global memory bandwidth consumption'
    ),
    'unique_arrays': (
        'Data transfer footprint',
        'Each distinct array requires a separate host↔device transfer (map clause)'
    ),
    'stride_one_ratio': (
        'Memory coalescing',
        'Stride-1 access lets a warp (32 threads) share one 128B memory transaction'
    ),
    'has_indirect_access': (
        'Scatter/gather',
        'A[B[i]] causes random memory access — destroys bandwidth efficiency'
    ),
    'data_reuse_score': (
        'Shared memory tiling opportunity',
        'High data reuse means GPU shared memory / L1 cache can be exploited'
    ),
    'estimated_data_bytes_log': (
        'PCIe transfer cost',
        'Bytes moved host↔device — the #1 GPU offload cost'
    ),
    'transfer_penalty': (
        'Transfer amortization',
        'Data transfer cost relative to computation — high penalty = transfer-dominated'
    ),
    'branch_count': (
        'Warp divergence',
        'Branches cause 32-thread warp to serialize both paths'
    ),
    'has_reduction': (
        'Reduction overhead',
        'Requires atomic or tree reduction — manageable but adds synchronization cost'
    ),
    'has_loop_carried_dep': (
        'Parallelism blocker',
        'Loop-carried dependency means iterations cannot run independently'
    ),
}

# Group features for SHAP explanation grouping
FEATURE_GROUPS = {
    # Parallelism Potential
    'trip_count_log': 'Parallelism Potential',
    'nest_depth': 'Parallelism Potential',
    'inner_loop_count': 'Parallelism Potential',
    # Compute Characteristics
    'total_ops': 'Compute Characteristics',
    'has_float_ops': 'Compute Characteristics',
    'comp_intensity': 'Compute Characteristics',
    # Memory Efficiency
    'mem_reads': 'Memory Efficiency',
    'mem_writes': 'Memory Efficiency',
    'unique_arrays': 'Memory Efficiency',
    'stride_one_ratio': 'Memory Efficiency',
    'has_indirect_access': 'Memory Efficiency',
    'data_reuse_score': 'Memory Efficiency',
    # Data Transfer Cost
    'estimated_data_bytes_log': 'Data Transfer Cost',
    'transfer_penalty': 'Data Transfer Cost',
    # Control Flow Overhead
    'branch_count': 'Control Flow Overhead',
    'has_reduction': 'Control Flow Overhead',
    'has_loop_carried_dep': 'Control Flow Overhead',
}


@dataclass
class DependencyInfo:
    """Dependency analysis result."""
    category: str    # "NONE" | "REDUCTION" | "DEPENDENCY" | "UNCERTAIN"
    details: str     # human-readable explanation
    accumulators: list = field(default_factory=list)  # [(var_name, op_str), ...]


@dataclass
class FeatureVector:
    """Complete feature extraction result for a single loop."""
    features: dict[str, float]
    loop_info: LoopInfo
    dependency: DependencyInfo
    blockers: list[dict]
    element_size: int                    # detected element size in bytes
    confidence_limited: bool = False     # True if unanalyzed function calls present
    confidence_reason: str = ""


# ---------------------------------------------------------------------------
# Main extraction entry point
# ---------------------------------------------------------------------------

def extract_features(loop_info: LoopInfo, source: bytes) -> FeatureVector:
    """
    Extract all 17 features from a parsed loop.

    Args:
        loop_info: LoopInfo from code_parser.
        source: Full source bytes for text extraction.

    Returns:
        FeatureVector with features dict, dependency info, blockers, etc.
    """
    body = loop_info.body_node
    if body is None:
        # Empty loop — return zero features
        return _empty_features(loop_info)

    # Detect offload blockers
    blockers = find_offload_blockers(body, source)

    # Detect element size from declarations
    element_size = _detect_element_size(loop_info.node, source)

    # --- Parallelism features ---
    trip_count_log = _compute_total_trip_count_log(loop_info, source)
    nest_depth = _compute_max_nesting_depth(body)
    inner_loop_count = _count_inner_loops(body)

    # --- Compute features ---
    total_ops = _count_arithmetic_ops(body, source)
    has_float_ops = _detect_float_ops(loop_info.node, source)
    mem_reads, mem_writes = _count_memory_ops(body, source)
    comp_intensity = total_ops / (mem_reads + mem_writes + 1)

    # --- Memory features ---
    arrays_read, arrays_written = _collect_array_names(body, source)
    all_arrays = arrays_read | arrays_written
    unique_arrays = len(all_arrays)
    stride_one_ratio = _compute_stride_one_ratio(
        body, source, loop_info.induction_var
    )
    has_indirect = _detect_indirect_access(body, source)
    reuse_score = _compute_data_reuse_score(
        body, source, loop_info.induction_var
    )

    # --- Transfer features ---
    total_trip = 2 ** trip_count_log
    
    # Fix the O(N^3) memory footprint bug!
    # If a loop has nesting and temporal reuse, the true memory footprint is vastly smaller than trip_count.
    # We estimate the 1D dimension N, and reduce the effective trip count by N^reuse_score.
    effective_trip_count = total_trip
    if nest_depth > 0 and reuse_score > 0:
        n_approx = total_trip ** (1.0 / (nest_depth + 1))
        effective_trip_count = total_trip / (n_approx ** min(1.0, reuse_score))
        
    est_data_bytes = unique_arrays * max(1, effective_trip_count) * element_size
    est_data_bytes_log = math.log2(max(est_data_bytes, 1))
    
    total_ops_scaled = total_ops * total_trip
    total_ops_log = math.log2(max(total_ops_scaled, 1))
    transfer_penalty = est_data_bytes_log - total_ops_log

    # --- Control flow features ---
    branch_count = _count_branches(body)
    dependency = _analyze_dependencies(body, source, loop_info.induction_var)
    has_reduction_flag = 1.0 if dependency.category == "REDUCTION" else 0.0
    has_dep_flag = 1.0 if dependency.category == "DEPENDENCY" else 0.0

    # --- Confidence limiting ---
    func_calls = _count_func_calls(body, source)
    has_early_exit = any(len(collect_nodes(body, t)) > 0 for t in ['break_statement', 'continue_statement', 'goto_statement', 'return_statement'])
    
    confidence_limited = func_calls > 0 or has_early_exit
    confidence_reason = ""
    if func_calls > 0:
        confidence_reason = (
            f"{func_calls} unanalyzed function call(s) in loop body - "
            "prediction confidence is limited"
        )
    if has_early_exit:
        if confidence_reason: confidence_reason += "; "
        confidence_reason += "data-dependent early exit (break/continue/goto/return) makes trip count and footprint unknowable statically"
        
    if dependency.category == "UNCERTAIN":
        confidence_limited = True
        if confidence_reason: confidence_reason += "; "
        confidence_reason += "dependency analysis inconclusive due to pointer/indirect patterns"

    features = {
        'trip_count_log': trip_count_log,
        'nest_depth': float(nest_depth),
        'inner_loop_count': float(inner_loop_count),
        'total_ops': float(total_ops),
        'has_float_ops': float(has_float_ops),
        'comp_intensity': round(comp_intensity, 4),
        'mem_reads': float(mem_reads),
        'mem_writes': float(mem_writes),
        'unique_arrays': float(unique_arrays),
        'stride_one_ratio': round(stride_one_ratio, 4),
        'has_indirect_access': float(has_indirect),
        'data_reuse_score': round(reuse_score, 4),
        'estimated_data_bytes_log': round(est_data_bytes_log, 4),
        'transfer_penalty': round(transfer_penalty, 4),
        'branch_count': float(branch_count),
        'has_reduction': has_reduction_flag,
        'has_loop_carried_dep': has_dep_flag,
    }

    return FeatureVector(
        features=features,
        loop_info=loop_info,
        dependency=dependency,
        blockers=blockers,
        element_size=element_size,
        confidence_limited=confidence_limited,
        confidence_reason=confidence_reason.strip('; '),
    )


# ---------------------------------------------------------------------------
# Parallelism features
# ---------------------------------------------------------------------------

def _compute_total_trip_count_log(loop_info: LoopInfo, source: bytes) -> float:
    """
    Compute log2 of the total iteration space (trip count).
    
    Recursively multiplies the candidate loop's bound by the bound of its
    nested loops. For sibling loops within the same body, we use max() 
    instead of sum(). 
    
    Modeling choice rationale: The static extractor's `total_ops` simply
    sums operations across the entire loop body. If we summed iterations of
    sequential sibling loops, `total_ops * sum(iters)` would wildly
    overestimate total work (effectively squaring the work). Using `max()` 
    maintains the correct algebraic invariant: O(N^k) structural scaling.
    """
    from src.code_parser import _parse_for_loop, _parse_while_loop, _parse_do_while_loop

    def get_node_bound(node: Node) -> tuple[int, Optional[Node]]:
        """Return (bound, body_node) for a loop node."""
        default_bound = 1000
        if node.type == 'for_statement':
            info = _parse_for_loop(node, source, 0, None)
        elif node.type == 'while_statement':
            info = _parse_while_loop(node, source, 0, None)
        elif node.type == 'do_statement':
            info = _parse_do_while_loop(node, source, 0, None)
        else:
            return 1, None
            
        bound = info.trip_count if (info and info.trip_count and info.trip_count > 0) else default_bound
        body = info.body_node if info else node.child_by_field_name('body')
        return bound, body

    def walk(node: Node) -> int:
        # Base case: no children, contributes nothing to multiplier
        if not node.children:
            return 1
            
        if node.type in ('for_statement', 'while_statement', 'do_statement'):
            bound, body = get_node_bound(node)
            if not body:
                return bound
            # Multiplicative property of nested loops
            return bound * walk(body)
        else:
            # Non-loop nodes: take the max multiplier of any child path.
            # KNOWN LIMITATION (Hackathon): For sequential sibling loops inside the 
            # same body (e.g. for(j<M){} for(k<K){}), this computes max(M, K) instead 
            # of (M + K). It undercounts back-to-back inner loops but perfectly 
            # handles single nested chains and if/else branching.
            return max((walk(child) for child in node.children), default=1)

    # Root loop
    root_bound = loop_info.trip_count if (loop_info.trip_count and loop_info.trip_count > 0) else 1000
    
    inner_max = 1
    if loop_info.body_node:
        inner_max = walk(loop_info.body_node)
            
    total_space = root_bound * inner_max
    return math.log2(max(total_space, 1))


def _count_inner_loops(body: Node) -> int:
    """Count for/while/do loops nested inside this body."""
    count = 0
    for node_type in ('for_statement', 'while_statement', 'do_statement'):
        count += count_node_type(body, node_type)
    return count


_LOOP_NODE_TYPES = frozenset(['for_statement', 'while_statement', 'do_statement'])


def _compute_max_nesting_depth(body: Node) -> int:
    """Compute max depth of loop nesting *inside* the candidate loop's body.

    Returns 0 for a flat loop with no inner loops, 1 for a single nested
    loop, 2 for triple-nested (GEMM outer → j → k), etc.

    This is NOT the same as loop_info.nesting_depth, which measures how
    deep this loop sits in its parent — always 0 for any loop that
    predictor.py actually analyzes (outermost/standalone candidates).
    """
    def walk(node: Node) -> int:
        if node.type in _LOOP_NODE_TYPES:
            # This is a loop inside the body — 1 + max depth of its own body
            loop_body = node.child_by_field_name('body')
            if loop_body:
                return 1 + walk(loop_body)
            return 1
        # Non-loop node: take the max over children
        return max((walk(child) for child in node.children), default=0)

    if body is None:
        return 0
    return walk(body)


# ---------------------------------------------------------------------------
# Compute features
# ---------------------------------------------------------------------------

_ARITH_OPS = frozenset(['+', '-', '*', '/', '%'])

# Compound assignment operators — tree-sitter-c parses x += y as a regular
# assignment_expression with the operator token as a child node whose .type
# is the literal operator string (e.g. "+="). There is NO separate
# "augmented_assignment_expression" node type in tree-sitter-c.
_COMPOUND_OPS = frozenset(['+=', '-=', '*=', '/=', '%=',
                           '<<=', '>>=', '&=', '|=', '^='])


def _is_compound_assignment(node: Node) -> bool:
    """Check if an assignment_expression uses a compound operator (+=, -= etc.).

    tree-sitter-c represents both `x = y` and `x += y` as
    assignment_expression nodes. The operator can be cleanly accessed
    via the 'operator' field.
    """
    if node.type != 'assignment_expression':
        return False
    op_node = node.child_by_field_name('operator')
    if op_node and op_node.type in _COMPOUND_OPS:
        return True
    return False


def _count_arithmetic_ops(body: Node, source: bytes) -> int:
    """
    Count arithmetic operations in the loop body.

    Counts:
      - binary_expression with operator +, -, *, /, %
      - update_expression (i++, i--)
      - compound assignment_expression (+=, -=, *=, /=, %=)
    """
    count = 0

    def walk(node: Node):
        nonlocal count
        if node.type == 'binary_expression':
            for child in node.children:
                if child.type in _ARITH_OPS:
                    count += 1
                    break
        elif node.type == 'update_expression':
            count += 1
        elif node.type == 'assignment_expression':
            if _is_compound_assignment(node):
                count += 1
        for child in node.children:
            walk(child)

    walk(body)
    return count


def _detect_float_ops(loop_node: Node, source: bytes) -> int:
    """
    Detect if loop body involves floating-point operations.

    Heuristic: checks for float/double declarations, .0 literals, f suffix.
    Returns 1 if float ops detected, 0 otherwise.
    """
    text = _node_text(loop_node, source).lower()

    # Check for type keywords
    if 'double' in text or 'float' in text:
        return 1

    # Check for float literals: 0.0, 1.0f, .5, 3.14
    import re
    if re.search(r'\d+\.\d*[fF]?\b', text):
        return 1
    if re.search(r'\.\d+[fF]?\b', text):
        return 1

    return 0


def _detect_element_size(
    loop_node: Node, source: bytes, default_element_size: int = 4
) -> int:
    """
    Estimate element size in bytes from type declarations.

    Scanning strategy (first match wins):
      1. The loop node text itself (catches in-loop declarations)
      2. The enclosing function_definition (catches parameter types like
         void f(double *A, ...) and local declarations above the loop)
      3. Fallback to default_element_size

    Known limitation: typedef aliases (e.g. PolyBench's DATA_TYPE) are
    opaque to text scanning.  The gcc -E preprocessing step expands
    #define DATA_TYPE double, which covers the macro case.  True C
    typedefs (typedef double DATA_TYPE;) remain unresolved — pass
    default_element_size=8 to override when the benchmark is known
    to use double.
    """
    # --- 1. Scan the loop node itself ---
    loop_text = _node_text(loop_node, source).lower()
    size = _type_text_to_size(loop_text)
    if size is not None:
        return size

    # --- 2. Walk up to enclosing function_definition ---
    parent = loop_node.parent
    while parent is not None:
        if parent.type == 'function_definition':
            func_text = _node_text(parent, source).lower()
            size = _type_text_to_size(func_text)
            if size is not None:
                return size
            break
        parent = parent.parent

    # --- 3. Fallback ---
    return default_element_size


def _type_text_to_size(text: str) -> Optional[int]:
    """Extract element size from source text containing type keywords.

    Returns the size in bytes, or None if no type keyword found.
    Priority order matches C type specificity.
    """
    # More specific types first to avoid substring false positives
    if 'long double' in text:
        return 16
    if 'double' in text:
        return 8
    if 'float' in text:
        return 4
    if 'long long' in text:
        return 8
    if 'short' in text:
        return 2
    if 'char' in text:
        return 1
    # int, long, unsigned — all 4 bytes on LP64
    return None


# ---------------------------------------------------------------------------
# Memory features
# ---------------------------------------------------------------------------

def _count_memory_ops(body: Node, source: bytes) -> tuple[int, int]:
    """
    Count array memory reads and writes in the loop body.

    Returns (reads, writes).
    An array subscript on the LHS of a simple assignment (=) is a write.
    An array subscript on the LHS of a compound assignment (+=, etc.) is
    both a read AND a write.
    All other array subscripts are reads.
    """
    reads = 0
    writes = 0

    def walk(node: Node, is_write_context: bool = False):
        nonlocal reads, writes

        if node.type == 'assignment_expression':
            left = node.child_by_field_name('left')
            right = node.child_by_field_name('right')
            is_compound = _is_compound_assignment(node)
            # LHS is write context
            if left:
                walk(left, is_write_context=True)
                # Compound assignment (+=, etc.) also reads the LHS
                if is_compound and left.type == 'subscript_expression':
                    reads += 1
            # RHS is read context
            if right:
                walk(right, is_write_context=False)
            return

        if node.type == 'update_expression':
            # x++ / x-- is a read-modify-write of its operand.
            # If the operand is a subscript (count[data[i]]++),
            # it counts as both a read and a write — same as +=.
            operand = node.child_by_field_name('argument')
            if operand is None:
                # Fallback: find non-operator child
                for child in node.children:
                    if child.type not in ('++', '--'):
                        operand = child
                        break
            if operand and operand.type == 'subscript_expression':
                reads += 1
                writes += 1
                # Walk the operand subscript's children, skipping argument-chaining
                arg_child = operand.child_by_field_name('argument')
                for child in operand.children:
                    if child == arg_child and child.type == 'subscript_expression':
                        continue
                    walk(child, is_write_context=False)
            elif operand:
                walk(operand, is_write_context)
            return

        if node.type == 'subscript_expression':
            if is_write_context:
                writes += 1
            else:
                reads += 1
            # Walk children, but distinguish multi-dim chaining from
            # index-position subscripts:
            #   A[i][j]        → argument field is inner subscript → SKIP (one access)
            #   count[data[i]] → index field has subscript → WALK (separate read)
            arg_child = node.child_by_field_name('argument')
            for child in node.children:
                if child == arg_child and child.type == 'subscript_expression':
                    continue  # multi-dim chain (A[i][j]) — already counted as one access
                walk(child, is_write_context=False)
            return

        for child in node.children:
            walk(child, is_write_context)

    walk(body)
    return reads, writes


def _collect_array_names(body: Node, source: bytes) -> tuple[set, set]:
    """
    Collect distinct array identifiers used in reads and writes.

    Returns (arrays_read, arrays_written).
    """
    arrays_read = set()
    arrays_written = set()

    def walk(node: Node, is_write: bool = False):
        if node.type == 'assignment_expression':
            left = node.child_by_field_name('left')
            right = node.child_by_field_name('right')
            is_compound = _is_compound_assignment(node)
            if left:
                walk(left, is_write=True)
                # Compound assignment (+=) also reads the LHS
                if is_compound:
                    walk(left, is_write=False)
            if right:
                walk(right, is_write=False)
            return

        if node.type == 'update_expression':
            operand = node.child_by_field_name('argument')
            if operand is None:
                for child in node.children:
                    if child.type not in ('++', '--'):
                        operand = child
                        break
            if operand:
                walk(operand, is_write=True)
                walk(operand, is_write=False)
            return

        if node.type == 'subscript_expression':
            # Get the array name (leftmost identifier in nested subscripts)
            arr_name = _get_array_base_name(node, source)
            if arr_name:
                if is_write:
                    arrays_written.add(arr_name)
                else:
                    arrays_read.add(arr_name)
            arg_child = node.child_by_field_name('argument')
            for child in node.children:
                if child == arg_child and child.type == 'subscript_expression':
                    continue
                walk(child, is_write=False)
            return

        for child in node.children:
            walk(child, is_write)

    walk(body)
    return arrays_read, arrays_written


def _get_array_base_name(subscript_node: Node, source: bytes) -> Optional[str]:
    """Get the base array name from a subscript expression (e.g., A from A[i][j])."""
    node = subscript_node
    # Walk left through nested subscript expressions
    while node.type == 'subscript_expression':
        arg = node.child_by_field_name('argument')
        if arg is None:
            # Fallback: first child
            children = [c for c in node.children if c.type != '[' and c.type != ']']
            if children:
                node = children[0]
            else:
                break
        else:
            node = arg
    if node.type == 'identifier':
        return _node_text(node, source)
    return None


def _compute_stride_one_ratio(
    body: Node, source: bytes, induction_var: Optional[str]
) -> float:
    """
    Compute fraction of array accesses that use the induction variable
    directly (stride-1 / coalesced access).

    A[i] → stride-1 (good for coalescing)
    A[i*N+j] → depends on context, treated as non-stride-1 conservatively
    A[expr_without_i] → not stride-1 but may be reuse (handled elsewhere)
    """
    if not induction_var:
        return 0.0

    subscripts = collect_nodes(body, 'subscript_expression')
    if not subscripts:
        return 0.0

    stride_one_count = 0
    total_count = 0

    for sub in subscripts:
        # Get the index expression (last child before ']')
        index_node = sub.child_by_field_name('index')
        if index_node is None:
            # Fallback: look for expression between [ and ]
            children = sub.children
            for i, child in enumerate(children):
                if child.type == '[' and i + 1 < len(children):
                    index_node = children[i + 1]
                    break
        if index_node is None:
            continue

        total_count += 1
        index_text = _node_text(index_node, source)

        # Stride-1: index is exactly the induction variable
        if index_text.strip() == induction_var:
            stride_one_count += 1
        # Also stride-1 if it's a simple offset: i+1, i-1
        elif _is_simple_offset(index_text, induction_var):
            stride_one_count += 1

    if total_count == 0:
        return 0.0
    return stride_one_count / total_count


def _is_simple_offset(index_text: str, induction_var: str) -> bool:
    """Check if index is induction_var ± small constant."""
    import re
    pattern = rf'^{re.escape(induction_var)}\s*[+-]\s*\d+$'
    return bool(re.match(pattern, index_text.strip()))


def _detect_indirect_access(body: Node, source: bytes) -> int:
    """
    Detect A[B[i]] pattern (nested subscript in index position).

    Returns 1 if indirect access found, 0 otherwise.
    """
    subscripts = collect_nodes(body, 'subscript_expression')
    for sub in subscripts:
        # Check if the index expression contains another subscript
        index_node = sub.child_by_field_name('index')
        if index_node is None:
            children = sub.children
            for i, child in enumerate(children):
                if child.type == '[' and i + 1 < len(children):
                    index_node = children[i + 1]
                    break
        if index_node and count_node_type(index_node, 'subscript_expression') > 0:
            return 1
    return 0


def _compute_data_reuse_score(
    body: Node, source: bytes, induction_var: Optional[str]
) -> float:
    """
    Estimate temporal/spatial data reuse in the loop nest.

    Scoring:
      +1 if array subscript is independent of induction variable (temporal reuse)
      +1 if stencil pattern detected (A[i-1], A[i], A[i+1])
      -1 if indirect access (anti-reuse)
       0 for streaming access (A[i])

    Returns normalized score: sum / total_accesses.
    Clamped to [-1.0, 1.0].
    """
    if not induction_var:
        return 0.0

    all_subscripts = collect_nodes(body, 'subscript_expression')
    if not all_subscripts:
        return 0.0

    # Filter to only outermost subscript expressions to handle multidimensional arrays properly
    outer_subscripts = []
    for sub in all_subscripts:
        if sub.parent and sub.parent.type == 'subscript_expression':
            continue
        outer_subscripts.append(sub)

    score = 0.0
    total = 0
    array_accesses: dict[str, list[str]] = {}

    for sub in outer_subscripts:
        total += 1
        arr_name = _get_array_base_name(sub, source)
        
        # Collect all index nodes across all dimensions (e.g., [i], [j] for A[i][j])
        index_nodes = []
        curr = sub
        while curr and curr.type == 'subscript_expression':
            idx = curr.child_by_field_name('index')
            if not idx:
                children = curr.children
                for i, child in enumerate(children):
                    if child.type == '[' and i + 1 < len(children):
                        idx = children[i + 1]
                        break
            if idx:
                index_nodes.append(idx)
            curr = curr.child_by_field_name('argument')

        if not index_nodes:
            continue

        index_texts = [_node_text(idx, source) for idx in index_nodes]

        if arr_name:
            array_accesses.setdefault(arr_name, []).extend(index_texts)

        # Check if the entire multi-dimensional access is independent of the induction variable
        import re
        has_induction = any(re.search(rf'\b{re.escape(induction_var)}\b', text) for text in index_texts)
        if not has_induction:
            # Independent of loop var → temporal reuse
            score += 1.0
        
        # Check for indirect access in any of the dimensions
        has_indirect = any(count_node_type(idx, 'subscript_expression') > 0 for idx in index_nodes)
        if has_indirect:
            # Indirect access → anti-reuse
            score -= 1.0

    # Bonus for stencil patterns
    for arr_name, indices in array_accesses.items():
        if len(indices) >= 2:
            # Check if we have offset patterns
            has_center = any(idx.strip() == induction_var for idx in indices)
            has_offset = any(
                _is_simple_offset(idx, induction_var) for idx in indices
            )
            if has_center and has_offset:
                score += 1.0  # stencil bonus

    if total == 0:
        return 0.0
    normalized = score / total
    return max(-1.0, min(1.0, normalized))


# ---------------------------------------------------------------------------
# Control flow features
# ---------------------------------------------------------------------------

def _count_branches(body: Node) -> int:
    """Count if/else/ternary/switch statements in the loop body."""
    count = 0
    count += count_node_type(body, 'if_statement')
    count += count_node_type(body, 'conditional_expression')  # ternary
    count += count_node_type(body, 'switch_statement')
    return count


def _count_func_calls(body: Node, source: bytes) -> int:
    """
    Count function calls in the loop body, excluding known-safe builtins.

    Calls to math functions (sin, cos, sqrt, etc.) are NOT counted as
    blocking — they have GPU intrinsic equivalents.
    """
    _SAFE_FUNCS = frozenset([
        'sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'atan2',
        'sqrt', 'cbrt', 'pow', 'exp', 'log', 'log2', 'log10',
        'fabs', 'abs', 'ceil', 'floor', 'round', 'fmin', 'fmax',
        'sinf', 'cosf', 'tanf', 'sqrtf', 'powf', 'expf', 'logf',
        'fabsf', 'fminf', 'fmaxf',
        'max', 'min',  # common macros
    ])

    calls = collect_nodes(body, 'call_expression')
    count = 0
    for call in calls:
        func_node = call.child_by_field_name('function')
        if func_node:
            func_name = _node_text(func_node, source)
            if func_name not in _SAFE_FUNCS:
                count += 1
    return count


def _analyze_dependencies(
    body: Node, source: bytes, induction_var: Optional[str]
) -> DependencyInfo:
    """
    Analyze loop-carried dependencies.

    Four-way classification:
      NONE       — no dependencies detected, safe to parallelize
      REDUCTION  — accumulation pattern (x += ..., x = max(x, ...))
      DEPENDENCY — loop-carried RAW dependency (a[i] = f(a[i-1]))
      UNCERTAIN  — can't determine (pointers, complex patterns)
    """
    if not induction_var or body is None:
        return DependencyInfo("UNCERTAIN", "No induction variable identified")

    # Collect all assignment targets and sources
    has_reduction = False
    has_dependency = False
    has_uncertain = False
    accumulators = []

    def get_compound_op(node: Node) -> str:
        op_node = node.child_by_field_name('operator')
        if op_node:
            t = _node_text(op_node, source)
            if t.endswith('='): return t[:-1]
        return '+'

    def get_simple_op(right: Node) -> str:
        if right.type == 'binary_expression':
            op_node = right.child_by_field_name('operator')
            if op_node:
                return _node_text(op_node, source)
        return '+'

    def walk(node: Node):
        nonlocal has_reduction, has_dependency, has_uncertain, accumulators

        if node.type == 'update_expression':
            operand = node.child_by_field_name('argument')
            if operand is None:
                for child in node.children:
                    if child.type not in ('++', '--'):
                        operand = child
                        break
            if operand and operand.type == 'identifier':
                operand_text = _node_text(operand, source)
                # Ignore the outer induction var, AND ignore typical inner ones (j, k)
                # or better: we should really collect them, but a quick heuristic is to ignore i, j, k, x, y, z
                if operand_text != induction_var and operand_text not in ('j', 'k', 'ii', 'jj', 'kk', 'x', 'y', 'z', 'idx'):
                    has_reduction = True
                    # assume + for ++ or --
                    accumulators.append((operand_text, '+'))

        if node.type == 'assignment_expression':
            left = node.child_by_field_name('left')
            right = node.child_by_field_name('right')
            is_compound = _is_compound_assignment(node)

            if left and right:
                left_text = _node_text(left, source)
                right_text = _node_text(right, source)

                if is_compound:
                    # Compound assignment: x += expr, A[i] += expr
                    if left.type == 'identifier' and left_text != induction_var:
                        # Scalar compound assignment → reduction (sum += ...)
                        has_reduction = True
                        accumulators.append((left_text, get_compound_op(node)))
                    elif left.type == 'subscript_expression':
                        # Array compound assignment: check for dependency
                        dep = _check_array_dependency(
                            left, right, source, induction_var
                        )
                        if dep == 'dependency':
                            has_dependency = True
                        elif dep == 'uncertain':
                            has_uncertain = True
                else:
                    # Simple assignment: x = x + expr → reduction check
                    if (left.type == 'identifier' and
                            left_text in right_text and
                            left_text != induction_var):
                        has_reduction = True
                        accumulators.append((left_text, get_simple_op(right)))

                    # Array: check if same array on both sides with offset
                    if left.type == 'subscript_expression':
                        dep = _check_array_dependency(
                            left, right, source, induction_var
                        )
                        if dep == 'dependency':
                            has_dependency = True
                        elif dep == 'uncertain':
                            has_uncertain = True

        # Pointer dereference on both LHS and RHS → uncertain
        if node.type == 'pointer_expression':
            has_uncertain = True

        for child in node.children:
            walk(child)

    walk(body)

    if has_dependency:
        return DependencyInfo(
            "DEPENDENCY",
            "Loop-carried dependency detected: array written at [i] "
            "depends on value at [i±k]"
        )
    if has_uncertain:
        return DependencyInfo(
            "UNCERTAIN",
            "Dependency analysis inconclusive — pointer or complex pattern detected"
        )
    if has_reduction:
        # deduplicate accumulators but preserve order
        unique_accs = []
        seen = set()
        for acc in accumulators:
            if acc not in seen:
                seen.add(acc)
                unique_accs.append(acc)
                
        return DependencyInfo(
            "REDUCTION",
            "Accumulation/reduction pattern detected — parallelizable "
            "with reduction clause",
            accumulators=unique_accs
        )
    return DependencyInfo("NONE", "No loop-carried dependencies detected")


def _check_array_dependency(
    lhs_sub: Node, rhs: Node, source: bytes, induction_var: str
) -> str:
    """
    Check if an array write depends on a different index of the same array.

    Returns: 'dependency', 'uncertain', or 'none'.
    """
    lhs_name = _get_array_base_name(lhs_sub, source)
    if not lhs_name:
        return 'none'

    # Collect all array accesses on RHS
    rhs_subscripts = collect_nodes(rhs, 'subscript_expression')
    for rhs_sub in rhs_subscripts:
        rhs_name = _get_array_base_name(rhs_sub, source)
        if rhs_name != lhs_name:
            continue

        # Same array on both sides — check if indices differ
        lhs_index = _get_subscript_index_text(lhs_sub, source)
        rhs_index = _get_subscript_index_text(rhs_sub, source)

        if lhs_index and rhs_index and lhs_index != rhs_index:
            # Same array, different index → loop-carried dependency
            # e.g., a[i] = a[i-1] + ...
            if induction_var in rhs_index:
                return 'dependency'
            else:
                return 'uncertain'

    return 'none'


def _get_subscript_index_text(sub: Node, source: bytes) -> Optional[str]:
    """Get the index expression text from a subscript expression."""
    index_node = sub.child_by_field_name('index')
    if index_node is None:
        children = sub.children
        for i, child in enumerate(children):
            if child.type == '[' and i + 1 < len(children):
                index_node = children[i + 1]
                break
    if index_node:
        return _node_text(index_node, source)
    return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _empty_features(loop_info: LoopInfo) -> FeatureVector:
    """Return a zero-valued feature vector for an empty/unparseable loop."""
    features = {name: 0.0 for name in FEATURE_NAMES}
    features['trip_count_log'] = _compute_total_trip_count_log(loop_info, b'')
    return FeatureVector(
        features=features,
        loop_info=loop_info,
        dependency=DependencyInfo("UNCERTAIN", "Empty or unparseable loop body"),
        blockers=[],
        element_size=4,
        confidence_limited=True,
        confidence_reason="Empty loop body",
    )


def features_to_list(features: dict[str, float]) -> list[float]:
    """Convert feature dict to list in canonical FEATURE_NAMES order."""
    return [features[name] for name in FEATURE_NAMES]
