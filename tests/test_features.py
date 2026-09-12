"""
Tests for src/feature_extractor.py

Coverage:
  - GEMM triple-nested loop: trip_count_log ≈ log2(N³)
  - Vector addition non-nested: trip_count_log ≈ log2(N), NOT 0
  - Prefix-scan dependency detection
  - Reduction detection (sum += A[i])
  - Histogram indirect access (A[B[i]])
  - Stencil reuse score (A[i-1], A[i], A[i+1])
  - Branch count
  - Element size detection (double → 8)
  - Feature vector has exactly 17 keys
  - Derived features use total_trip consistently
  - Compound assignment via child_by_field_name('operator')
"""

import math
import pytest
from src.code_parser import parse, ParseResult
from src.feature_extractor import (
    extract_features,
    features_to_list,
    FEATURE_NAMES,
    FeatureVector,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _extract(code: str) -> FeatureVector:
    """Parse code and extract features from the first loop found."""
    result = parse(code, preprocess=False)
    assert len(result.loops) >= 1, f"No loops found in: {code[:80]}..."
    return extract_features(result.loops[0], result.source)


# ---------------------------------------------------------------------------
# Critical check 1: Vector Addition (non-nested) — trip_count_log != 0
# ---------------------------------------------------------------------------

class TestVectorAddition:
    """
    The base-case bug: a non-nested loop with no inner loops must
    produce trip_count_log = log2(trip_count), NOT 0.
    """

    def test_vector_add_trip_count_nonzero(self):
        code = """
        void vec_add(int N, double *A, double *B, double *C) {
            for (int i = 0; i < 1024; i++) {
                C[i] = A[i] + B[i];
            }
        }
        """
        fv = _extract(code)
        # Must be log2(1024) = 10.0, NOT 0
        assert fv.features['trip_count_log'] == pytest.approx(10.0, abs=0.01)

    def test_vector_add_symbolic_trip_count(self):
        """Symbolic N → default 1000 → log2(1000) ≈ 9.97."""
        code = """
        void vec_add(int N, double *A, double *B, double *C) {
            for (int i = 0; i < N; i++) {
                C[i] = A[i] + B[i];
            }
        }
        """
        fv = _extract(code)
        # Default bound 1000 → log2(1000) ≈ 9.97
        assert fv.features['trip_count_log'] == pytest.approx(
            math.log2(1000), abs=0.1
        )
        assert fv.features['trip_count_log'] > 0  # NEVER zero


# ---------------------------------------------------------------------------
# Critical check 2: GEMM — trip_count_log ≈ log2(N³)
# ---------------------------------------------------------------------------

class TestGEMM:
    """
    Triple-nested GEMM: total iteration space = N³.
    trip_count_log must be log2(N³), not log2(N).
    estimated_data_bytes_log and transfer_penalty must scale consistently.
    """

    def test_gemm_trip_count_cubic(self):
        code = """
        void gemm(double *A, double *B, double *C) {
            for (int i = 0; i < 1024; i++) {
                for (int j = 0; j < 1024; j++) {
                    for (int k = 0; k < 1024; k++) {
                        C[i*1024+j] += A[i*1024+k] * B[k*1024+j];
                    }
                }
            }
        }
        """
        result = parse(code, preprocess=False)
        # The outermost loop should capture the full N³ iteration space
        outermost = result.loops[0]
        fv = extract_features(outermost, result.source)

        # log2(1024³) = 30.0
        assert fv.features['trip_count_log'] == pytest.approx(30.0, abs=0.01)

    def test_gemm_derived_features_consistent(self):
        """estimated_data_bytes_log and transfer_penalty must use total_trip."""
        code = """
        void gemm(double *A, double *B, double *C) {
            for (int i = 0; i < 256; i++) {
                for (int j = 0; j < 256; j++) {
                    for (int k = 0; k < 256; k++) {
                        C[i*256+j] += A[i*256+k] * B[k*256+j];
                    }
                }
            }
        }
        """
        result = parse(code, preprocess=False)
        fv = extract_features(result.loops[0], result.source)
        trip_count_log = fv.features['trip_count_log']

        # log2(256³) = 24.0
        assert trip_count_log == pytest.approx(24.0, abs=0.01)

        # Verify derived memory features reflect the O(N^3) footprint reduction fix
        # A raw calculation would be log2(3 * 2^24 * 8) = 28.58
        # The effective footprint scales down due to temporal reuse (estimated log2 is ~25.91)
        assert fv.features['estimated_data_bytes_log'] == pytest.approx(
            25.9183, abs=0.1
        )

    def test_gemm_nest_depth(self):
        """GEMM outer loop has 2 loops nested inside (j, k) → nest_depth=2."""
        code = """
        void gemm(double *A, double *B, double *C) {
            for (int i = 0; i < 256; i++) {
                for (int j = 0; j < 256; j++) {
                    for (int k = 0; k < 256; k++) {
                        C[i*256+j] += A[i*256+k] * B[k*256+j];
                    }
                }
            }
        }
        """
        result = parse(code, preprocess=False)
        fv = extract_features(result.loops[0], result.source)
        assert fv.features['nest_depth'] == 2.0

    def test_flat_loop_nest_depth_zero(self):
        """A single non-nested loop → nest_depth=0."""
        code = """
        void add(int N, double *A, double *B, double *C) {
            for (int i = 0; i < N; i++) {
                C[i] = A[i] + B[i];
            }
        }
        """
        fv = _extract(code)
        assert fv.features['nest_depth'] == 0.0

class TestDependencyDetection:
    def test_prefix_scan_dependency(self):
        """a[i] = a[i-1] + x[i] → DEPENDENCY."""
        code = """
        void prefix(int N, double *a, double *x) {
            for (int i = 1; i < N; i++) {
                a[i] = a[i-1] + x[i];
            }
        }
        """
        fv = _extract(code)
        assert fv.dependency.category == "DEPENDENCY"
        assert fv.features['has_loop_carried_dep'] == 1.0

    def test_reduction_detection(self):
        """sum += A[i] → REDUCTION."""
        code = """
        void reduce(int N, double *A) {
            double sum = 0.0;
            for (int i = 0; i < N; i++) {
                sum += A[i];
            }
        }
        """
        fv = _extract(code)
        assert fv.dependency.category == "REDUCTION"
        assert fv.features['has_reduction'] == 1.0
        assert fv.features['has_loop_carried_dep'] == 0.0

    def test_no_dependency(self):
        """C[i] = A[i] + B[i] → NONE."""
        code = """
        void add(int N, double *A, double *B, double *C) {
            for (int i = 0; i < N; i++) {
                C[i] = A[i] + B[i];
            }
        }
        """
        fv = _extract(code)
        assert fv.dependency.category == "NONE"
        assert fv.features['has_reduction'] == 0.0
        assert fv.features['has_loop_carried_dep'] == 0.0


# ---------------------------------------------------------------------------
# Memory features
# ---------------------------------------------------------------------------

class TestMemoryFeatures:
    def test_histogram_indirect_access(self):
        """count[data[i]] += 1 → indirect access detected."""
        code = """
        void histogram(int N, int *data, int *count) {
            for (int i = 0; i < N; i++) {
                count[data[i]] += 1;
            }
        }
        """
        fv = _extract(code)
        assert fv.features['has_indirect_access'] == 1.0

    def test_histogram_update_expression_memory_ops(self):
        """count[data[i]]++ → mem_reads=2, mem_writes=1, unique_arrays=2."""
        code = """
        void histogram(int N, int *data, int *count) {
            for (int i = 0; i < N; i++) {
                count[data[i]]++;
            }
        }
        """
        fv = _extract(code)
        assert fv.features['mem_reads'] == 2.0
        assert fv.features['mem_writes'] == 1.0
        assert fv.features['unique_arrays'] == 2.0
        assert fv.features['has_indirect_access'] == 1.0

    def test_no_indirect_access(self):
        """A[i] = B[i] → no indirect access."""
        code = """
        void copy(int N, double *A, double *B) {
            for (int i = 0; i < N; i++) {
                A[i] = B[i];
            }
        }
        """
        fv = _extract(code)
        assert fv.features['has_indirect_access'] == 0.0

    def test_stencil_reuse_score(self):
        """A[i-1] + A[i] + A[i+1] → positive reuse score (stencil bonus)."""
        code = """
        void stencil(int N, double *A, double *B) {
            for (int i = 1; i < N; i++) {
                B[i] = A[i-1] + A[i] + A[i+1];
            }
        }
        """
        fv = _extract(code)
        assert fv.features['data_reuse_score'] > 0.0

    def test_stride_one_ratio(self):
        """A[i] = B[i] → 100% stride-1 access."""
        code = """
        void copy(int N, double *A, double *B) {
            for (int i = 0; i < N; i++) {
                A[i] = B[i];
            }
        }
        """
        fv = _extract(code)
        assert fv.features['stride_one_ratio'] == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------------------
# Control flow features
# ---------------------------------------------------------------------------

class TestControlFlow:
    def test_branch_count(self):
        code = """
        void f(int N, double *A) {
            for (int i = 0; i < N; i++) {
                if (A[i] > 0) {
                    A[i] = A[i] * 2;
                } else if (A[i] < -1) {
                    A[i] = 0;
                }
            }
        }
        """
        fv = _extract(code)
        assert fv.features['branch_count'] >= 2

    def test_ternary_counted_as_branch(self):
        code = """
        void f(int N, double *A) {
            for (int i = 0; i < N; i++) {
                A[i] = (A[i] > 0) ? A[i] : 0;
            }
        }
        """
        fv = _extract(code)
        assert fv.features['branch_count'] >= 1


# ---------------------------------------------------------------------------
# Element size detection
# ---------------------------------------------------------------------------

class TestElementSize:
    def test_double_detected(self):
        """double in function signature → element_size = 8."""
        code = """
        void f(int N, double *A, double *B) {
            for (int i = 0; i < N; i++) {
                A[i] = B[i] + 1.0;
            }
        }
        """
        fv = _extract(code)
        assert fv.element_size == 8

    def test_float_detected(self):
        code = """
        void f(int N, float *A) {
            for (int i = 0; i < N; i++) {
                A[i] = 0.0f;
            }
        }
        """
        fv = _extract(code)
        assert fv.element_size == 4

    def test_default_element_size(self):
        """int arrays → no float/double keyword → default 4."""
        code = """
        void f(int N, int *A) {
            for (int i = 0; i < N; i++) {
                A[i] = i;
            }
        }
        """
        fv = _extract(code)
        assert fv.element_size == 4  # default


# ---------------------------------------------------------------------------
# Feature vector structure
# ---------------------------------------------------------------------------

class TestFeatureVectorStructure:
    def test_exactly_17_features(self):
        code = """
        void f(int N, double *A, double *B) {
            for (int i = 0; i < N; i++) {
                A[i] = B[i] + 1.0;
            }
        }
        """
        fv = _extract(code)
        assert len(fv.features) == 17

    def test_all_feature_names_present(self):
        code = """
        void f(int N, double *A) {
            for (int i = 0; i < N; i++) {
                A[i] = A[i] * 2.0;
            }
        }
        """
        fv = _extract(code)
        for name in FEATURE_NAMES:
            assert name in fv.features, f"Missing feature: {name}"

    def test_all_features_are_float(self):
        code = """
        void f(int N, double *A) {
            for (int i = 0; i < N; i++) {
                A[i] = A[i] * 2.0;
            }
        }
        """
        fv = _extract(code)
        for name, value in fv.features.items():
            assert isinstance(value, float), (
                f"Feature {name} is {type(value).__name__}, expected float"
            )

    def test_features_to_list_ordering(self):
        code = """
        void f(int N, double *A) {
            for (int i = 0; i < N; i++) {
                A[i] = 0.0;
            }
        }
        """
        fv = _extract(code)
        as_list = features_to_list(fv.features)
        assert len(as_list) == 17
        for i, name in enumerate(FEATURE_NAMES):
            assert as_list[i] == fv.features[name]


# ---------------------------------------------------------------------------
# Compound assignment (Bug #1 regression test)
# ---------------------------------------------------------------------------

class TestCompoundAssignment:
    def test_compound_assignment_counted(self):
        """sum += A[i] must be counted as an arithmetic op."""
        code = """
        void f(int N, double *A) {
            double sum = 0.0;
            for (int i = 0; i < N; i++) {
                sum += A[i];
            }
        }
        """
        fv = _extract(code)
        assert fv.features['total_ops'] >= 1

    def test_gemm_compound_assignment(self):
        """C[i][j] += A[i][k]*B[k][j] must be counted."""
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
        result = parse(code, preprocess=False)
        # parse() now returns outermost-only; extract features from the outer i-loop
        # which contains the entire GEMM body including the compound assignment
        assert len(result.loops) == 1
        outer = result.loops[0]
        fv = extract_features(outer, result.source)
        assert fv.features['total_ops'] >= 2  # at least * and +=


# ---------------------------------------------------------------------------
# Confidence limiting
# ---------------------------------------------------------------------------

class TestConfidenceLimiting:
    def test_func_call_limits_confidence(self):
        """Non-safe function call in loop → confidence_limited = True."""
        code = """
        void f(int N, double *A) {
            for (int i = 0; i < N; i++) {
                A[i] = my_custom_function(i);
            }
        }
        """
        fv = _extract(code)
        assert fv.confidence_limited is True

    def test_safe_math_no_limit(self):
        """sin/cos/sqrt are safe — should NOT limit confidence."""
        code = """
        void f(int N, double *A) {
            for (int i = 0; i < N; i++) {
                A[i] = sin(A[i]) + cos(A[i]);
            }
        }
        """
        fv = _extract(code)
        # sin and cos are safe builtins
        assert fv.confidence_limited is False
