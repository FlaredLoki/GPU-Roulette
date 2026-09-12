"""
tree-sitter based C code parser for GPUGate.

Responsibilities:
- Parse raw C code into a tree-sitter CST
- Identify candidate loop constructs (for, while)
- Extract loop bounds to estimate trip count
- Detect OpenMP/OpenACC pragmas
- Provide AST walking utilities for feature extraction

Requires: pip install tree-sitter tree-sitter-c
Optional: gcc -E preprocessing for macro expansion
"""

import subprocess
import tempfile
import os
import re
from dataclasses import dataclass, field
from typing import Optional

import tree_sitter_c as tsc
from tree_sitter import Language, Parser, Node

# Initialize tree-sitter C language and parser (module-level singleton)
C_LANGUAGE = Language(tsc.language())
_parser = Parser(C_LANGUAGE)


@dataclass
class LoopInfo:
    """Parsed information about a candidate loop."""
    node: Node                          # tree-sitter AST node for the loop
    loop_type: str                      # "for" | "while" | "do_while"
    line: int                           # 1-indexed source line number
    column: int                         # 0-indexed column
    nesting_depth: int                  # 0 = outermost
    trip_count: Optional[int]           # estimated static trip count, None if unknown
    trip_count_symbol: Optional[str]    # symbolic bound name (e.g. "N"), None if literal
    induction_var: Optional[str]        # loop induction variable name (e.g. "i")
    body_node: Optional[Node]           # AST node for the loop body
    parent_loop: Optional['LoopInfo']   # enclosing loop, None if outermost
    source_text: str                    # raw source text of the loop


@dataclass
class ParseResult:
    """Result of parsing a C source string."""
    tree: object                        # tree-sitter Tree
    source: bytes                       # the (possibly preprocessed) source bytes
    loops: list[LoopInfo] = field(default_factory=list)
    pragmas: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess_source(code: str, gcc_path: str = "gcc") -> tuple[str, bool]:
    """
    Run gcc -E to expand macros. Falls back to raw code if gcc unavailable.

    Returns (processed_code, was_preprocessed).
    """
    try:
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.c', delete=False, encoding='utf-8'
        ) as f:
            f.write(code)
            tmp_path = f.name

        result = subprocess.run(
            [gcc_path, "-E", "-P",  # -P suppresses line markers
             "-xc",                 # treat input as C
             tmp_path],
            capture_output=True, text=True, timeout=10
        )
        os.unlink(tmp_path)

        if result.returncode == 0 and result.stdout.strip():
            return result.stdout, True
        else:
            return code, False

    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        # gcc not available or timed out — fall back to raw source
        try:
            os.unlink(tmp_path)
        except (NameError, OSError):
            pass
        return code, False


# ---------------------------------------------------------------------------
# Core parsing
# ---------------------------------------------------------------------------

def parse(code: str, preprocess: bool = True) -> ParseResult:
    """
    Parse C source code and extract candidate loops.

    Args:
        code: C source code string.
        preprocess: If True, attempt gcc -E macro expansion first.

    Returns:
        ParseResult with tree, loops, pragmas, and any errors.
    """
    preprocessed = False
    if preprocess:
        code, preprocessed = preprocess_source(code)

    source_bytes = code.encode('utf-8')
    tree = _parser.parse(source_bytes)
    result = ParseResult(tree=tree, source=source_bytes)

    if not preprocessed and preprocess:
        result.errors.append(
            "Macros not expanded (gcc not found). "
            "Some features may be approximate."
        )

    # Check for parse errors
    if tree.root_node.has_error:
        result.errors.append("Source contains syntax errors; analysis may be incomplete.")

    # Extract pragmas before loop detection
    result.pragmas = _extract_pragmas(tree.root_node, source_bytes)

    # Find all candidate loops
    all_loops = _find_candidate_loops(tree.root_node, source_bytes)
    
    # We only expose outermost loops as prediction candidates. 
    # Nested loops are inherently analyzed as part of their parent's body.
    result.loops = [loop for loop in all_loops if loop.parent_loop is None]

    return result


# ---------------------------------------------------------------------------
# Loop detection
# ---------------------------------------------------------------------------

_OFFLOAD_BLOCKERS = frozenset([
    # I/O functions
    'printf', 'fprintf', 'scanf', 'fscanf', 'puts', 'gets', 'fgets',
    'fopen', 'fclose', 'fread', 'fwrite', 'fseek',
    # Memory management (dynamic alloc in kernel = UB)
    'malloc', 'calloc', 'realloc', 'free',
    # Process/thread management
    'fork', 'exec', 'pthread_create', 'exit', 'abort',
    # C++ new/delete caught by keyword, not function name
])


def _find_candidate_loops(
    root: Node,
    source: bytes,
    depth: int = 0,
    parent: Optional[LoopInfo] = None
) -> list[LoopInfo]:
    """Recursively find all loop constructs in the AST."""
    loops = []

    for child in root.children:
        if child.type == 'for_statement':
            info = _parse_for_loop(child, source, depth, parent)
            loops.append(info)
            # Recurse into body for nested loops
            if info.body_node:
                nested = _find_candidate_loops(
                    info.body_node, source, depth + 1, info
                )
                loops.extend(nested)

        elif child.type == 'while_statement':
            info = _parse_while_loop(child, source, depth, parent)
            loops.append(info)
            if info.body_node:
                nested = _find_candidate_loops(
                    info.body_node, source, depth + 1, info
                )
                loops.extend(nested)

        elif child.type == 'do_statement':
            info = _parse_do_while_loop(child, source, depth, parent)
            loops.append(info)
            if info.body_node:
                nested = _find_candidate_loops(
                    info.body_node, source, depth + 1, info
                )
                loops.extend(nested)

        else:
            # Recurse into non-loop compound statements (functions, blocks, etc.)
            nested = _find_candidate_loops(child, source, depth, parent)
            loops.extend(nested)

    return loops


def _parse_for_loop(
    node: Node, source: bytes, depth: int, parent: Optional[LoopInfo]
) -> LoopInfo:
    """Parse a for_statement node and extract loop metadata."""
    # tree-sitter for_statement children:
    #   "for" "(" initializer ";" condition ";" update ")" body
    induction_var = None
    trip_count = None
    trip_count_symbol = None
    body_node = None

    # Find the initializer, condition, update, and body
    # For statement has named children in tree-sitter-c:
    #   initializer, condition, update, body (as a compound_statement)
    init_node = None
    cond_node = None
    update_node = None

    # Walk children to find components
    # tree-sitter-c for_statement structure:
    # (for_statement
    #   "for" "("
    #   initializer: (declaration / expression_statement / ...)
    #   condition: (binary_expression / ...)
    #   update: (update_expression / ...)
    #   body: (compound_statement / ...))
    for child in node.children:
        if child.type == 'compound_statement' or (
            child.type not in ('for', '(', ')', ';') and
            child == node.children[-1]
        ):
            body_node = child
        # Use field names when available
    
    # Use tree-sitter field API for reliable extraction
    init_node = node.child_by_field_name('initializer')
    cond_node = node.child_by_field_name('condition')
    update_node = node.child_by_field_name('update')
    body_node = node.child_by_field_name('body') or body_node

    # Extract induction variable from initializer
    if init_node:
        induction_var = _extract_induction_var(init_node, source)

    # Estimate trip count from condition
    if cond_node and induction_var:
        trip_count, trip_count_symbol = _estimate_trip_count(
            cond_node, induction_var, source
        )

    return LoopInfo(
        node=node,
        loop_type="for",
        line=node.start_point[0] + 1,  # 1-indexed
        column=node.start_point[1],
        nesting_depth=depth,
        trip_count=trip_count,
        trip_count_symbol=trip_count_symbol,
        induction_var=induction_var,
        body_node=body_node,
        parent_loop=parent,
        source_text=_node_text(node, source),
    )


def _parse_while_loop(
    node: Node, source: bytes, depth: int, parent: Optional[LoopInfo]
) -> LoopInfo:
    """Parse a while_statement — trip count always unknown."""
    body_node = node.child_by_field_name('body')
    return LoopInfo(
        node=node,
        loop_type="while",
        line=node.start_point[0] + 1,
        column=node.start_point[1],
        nesting_depth=depth,
        trip_count=None,
        trip_count_symbol=None,
        induction_var=None,
        body_node=body_node,
        parent_loop=parent,
        source_text=_node_text(node, source),
    )


def _parse_do_while_loop(
    node: Node, source: bytes, depth: int, parent: Optional[LoopInfo]
) -> LoopInfo:
    """Parse a do_statement — trip count always unknown."""
    body_node = node.child_by_field_name('body')
    return LoopInfo(
        node=node,
        loop_type="do_while",
        line=node.start_point[0] + 1,
        column=node.start_point[1],
        nesting_depth=depth,
        trip_count=None,
        trip_count_symbol=None,
        induction_var=None,
        body_node=body_node,
        parent_loop=parent,
        source_text=_node_text(node, source),
    )


# ---------------------------------------------------------------------------
# Trip count estimation
# ---------------------------------------------------------------------------

def _extract_induction_var(init_node: Node, source: bytes) -> Optional[str]:
    """
    Extract the induction variable name from a for-loop initializer.

    Handles:
        int i = 0       → "i"
        i = 0           → "i"
        int i = start   → "i"
    """
    # Case 1: declaration — "int i = 0"
    if init_node.type == 'declaration':
        declarator = init_node.child_by_field_name('declarator')
        if declarator:
            # init_declarator → declarator (identifier) "=" value
            if declarator.type == 'init_declarator':
                name_node = declarator.child_by_field_name('declarator')
                if name_node and name_node.type == 'identifier':
                    return _node_text(name_node, source)
            elif declarator.type == 'identifier':
                return _node_text(declarator, source)

    # Case 2: expression_statement or plain assignment — "i = 0"
    # Walk to find an assignment_expression
    assign = _find_first(init_node, 'assignment_expression')
    if assign:
        left = assign.child_by_field_name('left')
        if left and left.type == 'identifier':
            return _node_text(left, source)

    # Case 3: just an identifier
    ident = _find_first(init_node, 'identifier')
    if ident:
        return _node_text(ident, source)

    return None


def _estimate_trip_count(
    cond_node: Node,
    induction_var: str,
    source: bytes
) -> tuple[Optional[int], Optional[str]]:
    """
    Estimate trip count from a for-loop condition.

    Handles:
        i < N       → (None, "N")      symbolic
        i < 1024    → (1024, None)      literal
        i <= N-1    → (None, "N")       symbolic
        i < N*N     → (None, "N*N")     symbolic expression

    Returns (trip_count_or_None, symbol_or_None).
    """
    # Look for a binary expression: induction_var <op> bound
    if cond_node.type != 'binary_expression':
        return None, None

    left = cond_node.child_by_field_name('left')
    op = cond_node.child_by_field_name('operator')
    right = cond_node.child_by_field_name('right')

    # Fallback: iterate children to find operator
    if op is None:
        children = cond_node.children
        for i, child in enumerate(children):
            if child.type in ('<', '<=', '>', '>=', '!='):
                op = child
                if i > 0:
                    left = children[i - 1]
                if i + 1 < len(children):
                    right = children[i + 1]
                break

    if left is None or right is None or op is None:
        return None, None

    left_text = _node_text(left, source)
    right_text = _node_text(right, source)
    op_text = _node_text(op, source)

    # Ensure left side is the induction variable
    if left_text != induction_var:
        # Maybe reversed: N > i
        if right_text == induction_var and op_text in ('>', '>='):
            left_text, right_text = right_text, left_text
            op_text = '<' if op_text == '>' else '<='
        else:
            return None, None

    # Try to parse right side as integer literal
    bound_value = _try_parse_int(right_text)

    if bound_value is not None:
        # Literal bound
        if op_text == '<':
            return bound_value, None
        elif op_text == '<=':
            return bound_value + 1, None
        elif op_text == '!=':
            return bound_value, None  # common for i != N
        else:
            return bound_value, None
    else:
        # Symbolic bound
        return None, right_text


def _try_parse_int(text: str) -> Optional[int]:
    """Try to parse a string as an integer literal."""
    text = text.strip()
    try:
        return int(text)
    except ValueError:
        pass
    # Handle hex
    if text.startswith('0x') or text.startswith('0X'):
        try:
            return int(text, 16)
        except ValueError:
            pass
    return None


# ---------------------------------------------------------------------------
# Pragma detection
# ---------------------------------------------------------------------------

def _extract_pragmas(root: Node, source: bytes) -> list[dict]:
    """Find OpenMP/OpenACC pragma directives in the source."""
    pragmas = []

    def walk(node: Node):
        if node.type == 'preproc_call':
            text = _node_text(node, source)
            if '#pragma' in text and ('omp' in text or 'acc' in text):
                pragma_type = 'omp' if 'omp' in text else 'acc'
                pragmas.append({
                    'type': pragma_type,
                    'text': text.strip(),
                    'line': node.start_point[0] + 1,
                    'is_target': 'target' in text,
                    'is_parallel': 'parallel' in text,
                    'has_map': 'map(' in text,
                })
        for child in node.children:
            walk(child)

    walk(root)
    return pragmas


# ---------------------------------------------------------------------------
# Offload-blocker detection
# ---------------------------------------------------------------------------

def find_offload_blockers(body_node: Node, source: bytes) -> list[dict]:
    """
    Find constructs in a loop body that prevent GPU offloading.

    Returns list of {reason, line, text} dicts.
    """
    blockers = []

    def walk(node: Node):
        if node.type == 'call_expression':
            func_node = node.child_by_field_name('function')
            if func_node:
                func_name = _node_text(func_node, source)
                if func_name in _OFFLOAD_BLOCKERS:
                    blockers.append({
                        'reason': _blocker_reason(func_name),
                        'line': node.start_point[0] + 1,
                        'text': func_name,
                    })
        for child in node.children:
            walk(child)

    if body_node:
        walk(body_node)
    return blockers


def _blocker_reason(func_name: str) -> str:
    """Return a human-readable reason for why a function blocks offloading."""
    if func_name in ('printf', 'fprintf', 'scanf', 'fscanf',
                      'puts', 'gets', 'fgets', 'fopen', 'fclose',
                      'fread', 'fwrite', 'fseek'):
        return f"I/O function '{func_name}' not supported on GPU"
    elif func_name in ('malloc', 'calloc', 'realloc', 'free'):
        return f"Dynamic memory allocation '{func_name}' in GPU kernel is undefined behavior"
    elif func_name in ('fork', 'exec', 'pthread_create'):
        return f"Process/thread management '{func_name}' invalid on GPU"
    elif func_name in ('exit', 'abort'):
        return f"Program termination '{func_name}' not supported on GPU"
    return f"Function '{func_name}' may not be available on GPU"


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _node_text(node: Node, source: bytes) -> str:
    """Extract the source text of an AST node."""
    return source[node.start_byte:node.end_byte].decode('utf-8', errors='replace')


def _find_first(node: Node, node_type: str) -> Optional[Node]:
    """Find the first descendant node of a given type (DFS)."""
    if node.type == node_type:
        return node
    for child in node.children:
        result = _find_first(child, node_type)
        if result:
            return result
    return None


def count_node_type(node: Node, node_type: str) -> int:
    """Count all descendant nodes of a given type."""
    count = 1 if node.type == node_type else 0
    for child in node.children:
        count += count_node_type(child, node_type)
    return count


def collect_nodes(node: Node, node_type: str) -> list[Node]:
    """Collect all descendant nodes of a given type."""
    results = []
    if node.type == node_type:
        results.append(node)
    for child in node.children:
        results.extend(collect_nodes(child, node_type))
    return results
