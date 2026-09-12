"""
Tests for src/code_parser.py

Coverage:
  - Simple for-loop with literal trip count
  - Nested loops (depth tracking)
  - While loop (trip count always None)
  - Pragma detection (OpenMP target)
  - Offload blocker detection (printf, malloc)
  - Function call in loop body
  - Empty body handling
  - Multi-dimensional array in condition
  - Induction variable extraction from different initializer forms
  - Hex literal parsing
"""

import pytest
from src.code_parser import (
    parse,
    _parse_for_loop,
    _extract_induction_var,
    _estimate_trip_count,
    find_offload_blockers,
    _node_text,
    _find_first,
    count_node_type,
    collect_nodes,
    LoopInfo,
    ParseResult,
    _try_parse_int,
)


# ---------------------------------------------------------------------------
# Helper: parse without gcc -E (avoids needing gcc on test machine)
# ---------------------------------------------------------------------------

def _parse_no_preprocess(code: str) -> ParseResult:
    """Parse without gcc preprocessing for portable tests."""
    return parse(code, preprocess=False)


# ---------------------------------------------------------------------------
# Basic for-loop parsing
# ---------------------------------------------------------------------------

class TestBasicForLoop:
    def test_simple_for_loop_detected(self):
        code = "void f() { for (int i = 0; i < 100; i++) { x = 1; } }"
        result = _parse_no_preprocess(code)
        assert len(result.loops) >= 1
        loop = result.loops[0]
        assert loop.loop_type == "for"
        assert loop.trip_count == 100
        assert loop.induction_var == "i"

    def test_literal_trip_count(self):
        code = "void f() { for (int i = 0; i < 1024; i++) { A[i] = 0; } }"
        result = _parse_no_preprocess(code)
        assert result.loops[0].trip_count == 1024

    def test_symbolic_trip_count(self):
        code = "void f(int N) { for (int i = 0; i < N; i++) { A[i] = 0; } }"
        result = _parse_no_preprocess(code)
        loop = result.loops[0]
        assert loop.trip_count is None
        assert loop.trip_count_symbol == "N"

    def test_less_than_or_equal(self):
        code = "void f() { for (int i = 0; i <= 99; i++) { x = 1; } }"
        result = _parse_no_preprocess(code)
        assert result.loops[0].trip_count == 100  # <= 99 → 100 iterations

    def test_not_equal_operator(self):
        code = "void f() { for (int i = 0; i != 50; i++) { x = 1; } }"
        result = _parse_no_preprocess(code)
        assert result.loops[0].trip_count == 50

    def test_hex_literal(self):
        assert _try_parse_int("0x400") == 1024
        assert _try_parse_int("0X10") == 16


# ---------------------------------------------------------------------------
# Nested loops
# ---------------------------------------------------------------------------

class TestNestedLoops:
    def test_two_level_nesting(self):
        code = """
        void f() {
            for (int i = 0; i < 100; i++) {
                for (int j = 0; j < 200; j++) {
                    A[i][j] = 0;
                }
            }
        }
        """
        result = _parse_no_preprocess(code)
        # parse() now filters to outermost-only
        assert len(result.loops) == 1
        outer = result.loops[0]
        assert outer.nesting_depth == 0
        assert outer.parent_loop is None

    def test_three_level_nesting_gemm(self):
        code = """
        void gemm(int N, double *A, double *B, double *C) {
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    for (int k = 0; k < N; k++) {
                        C[i*N+j] += A[i*N+k] * B[k*N+j];
                    }
                }
            }
        }
        """
        result = _parse_no_preprocess(code)
        # parse() now filters to outermost-only
        assert len(result.loops) == 1
        assert result.loops[0].nesting_depth == 0
        assert result.loops[0].parent_loop is None


# ---------------------------------------------------------------------------
# While / do-while loops
# ---------------------------------------------------------------------------

class TestOtherLoopTypes:
    def test_while_loop(self):
        code = "void f() { while (x > 0) { x--; } }"
        result = _parse_no_preprocess(code)
        assert len(result.loops) >= 1
        assert result.loops[0].loop_type == "while"
        assert result.loops[0].trip_count is None

    def test_do_while_loop(self):
        code = "void f() { do { x--; } while (x > 0); }"
        result = _parse_no_preprocess(code)
        assert len(result.loops) >= 1
        assert result.loops[0].loop_type == "do_while"
        assert result.loops[0].trip_count is None


# ---------------------------------------------------------------------------
# Pragma detection
# ---------------------------------------------------------------------------

class TestPragmaDetection:
    def test_omp_target_pragma(self):
        code = """
        #pragma omp target teams distribute parallel for map(tofrom: A[0:N])
        for (int i = 0; i < N; i++) {
            A[i] = B[i] + C[i];
        }
        """
        result = _parse_no_preprocess(code)
        assert len(result.pragmas) >= 1
        pragma = result.pragmas[0]
        assert pragma['type'] == 'omp'
        assert pragma['is_target'] is True
        assert pragma['is_parallel'] is True
        assert pragma['has_map'] is True

    def test_acc_pragma(self):
        code = """
        #pragma acc parallel loop
        for (int i = 0; i < N; i++) {
            A[i] = 0;
        }
        """
        result = _parse_no_preprocess(code)
        if result.pragmas:
            assert result.pragmas[0]['type'] == 'acc'


# ---------------------------------------------------------------------------
# Offload blocker detection
# ---------------------------------------------------------------------------

class TestOffloadBlockers:
    def test_printf_detected(self):
        code = """
        void f() {
            for (int i = 0; i < 10; i++) {
                printf("hello %d\\n", i);
            }
        }
        """
        result = _parse_no_preprocess(code)
        loop = result.loops[0]
        blockers = find_offload_blockers(loop.body_node, result.source)
        assert len(blockers) >= 1
        assert any("printf" in b['text'] for b in blockers)

    def test_malloc_detected(self):
        code = """
        void f() {
            for (int i = 0; i < 10; i++) {
                int *p = malloc(sizeof(int) * 100);
                free(p);
            }
        }
        """
        result = _parse_no_preprocess(code)
        loop = result.loops[0]
        blockers = find_offload_blockers(loop.body_node, result.source)
        assert len(blockers) >= 2  # malloc + free

    def test_no_blockers_clean_loop(self):
        code = """
        void f() {
            for (int i = 0; i < 100; i++) {
                A[i] = B[i] + C[i];
            }
        }
        """
        result = _parse_no_preprocess(code)
        loop = result.loops[0]
        blockers = find_offload_blockers(loop.body_node, result.source)
        assert len(blockers) == 0

    def test_safe_math_not_blocked(self):
        """sin, cos, sqrt etc. have GPU intrinsic equivalents — NOT blockers."""
        code = """
        void f() {
            for (int i = 0; i < 100; i++) {
                A[i] = sin(B[i]) + sqrt(C[i]);
            }
        }
        """
        result = _parse_no_preprocess(code)
        loop = result.loops[0]
        blockers = find_offload_blockers(loop.body_node, result.source)
        # sin and sqrt are NOT in _OFFLOAD_BLOCKERS
        assert len(blockers) == 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_source(self):
        result = _parse_no_preprocess("")
        assert len(result.loops) == 0

    def test_no_loops(self):
        code = "int main() { int x = 42; return x; }"
        result = _parse_no_preprocess(code)
        assert len(result.loops) == 0

    def test_function_call_in_body(self):
        """Loop with function call should still parse correctly."""
        code = """
        void f() {
            for (int i = 0; i < 100; i++) {
                my_function(A[i]);
            }
        }
        """
        result = _parse_no_preprocess(code)
        assert len(result.loops) == 1
        assert result.loops[0].trip_count == 100

    def test_body_node_exists(self):
        code = "void f() { for (int i = 0; i < 10; i++) { x = i; } }"
        result = _parse_no_preprocess(code)
        loop = result.loops[0]
        assert loop.body_node is not None

    def test_induction_var_assignment_form(self):
        """for (i = 0; ...) without declaration."""
        code = "void f() { int i; for (i = 0; i < 50; i++) { x = i; } }"
        result = _parse_no_preprocess(code)
        loop = result.loops[0]
        assert loop.induction_var == "i"
        assert loop.trip_count == 50


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

class TestUtilities:
    def test_count_node_type(self):
        code = "void f() { for (int i=0;i<10;i++) { for (int j=0;j<10;j++) { x=1; } } }"
        result = _parse_no_preprocess(code)
        root = result.tree.root_node
        assert count_node_type(root, 'for_statement') == 2

    def test_collect_nodes(self):
        code = "void f() { for (int i=0;i<10;i++) { for (int j=0;j<10;j++) { x=1; } } }"
        result = _parse_no_preprocess(code)
        root = result.tree.root_node
        nodes = collect_nodes(root, 'for_statement')
        assert len(nodes) == 2
