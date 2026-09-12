"""
tests/test_predictor.py

Integration tests for the end-to-end predictor pipeline.
"""
import pytest
import json
from src import predictor
from dataclasses import asdict

# Assuming model.py provides a mock when there's no trained model

@pytest.fixture
def pred():
    return predictor.GPUGatePredictor()

def test_empty_code(pred):
    results, _ = pred.predict("")
    assert len(results) == 0

def test_blocked_prediction(pred):
    code = """
    void foo(int N) {
        for (int i = 0; i < N; i++) {
            printf("%d", i);
        }
    }
    """
    results, _ = pred.predict(code)
    assert len(results) == 1
    assert results[0].verdict == "BLOCKED"
    assert results[0].confidence == 1.0
    assert any("printf" in str(b) for b in results[0].blockers)

def test_dependency_detected(pred):
    code = """
    void prefix(int N, double *A) {
        for (int i = 1; i < N; i++) {
            A[i] = A[i] + A[i-1];
        }
    }
    """
    results, _ = pred.predict(code)
    assert len(results) == 1
    assert results[0].verdict == "UNPROFITABLE"
    assert results[0].dependency is not None
    assert results[0].dependency.category == "DEPENDENCY"
    assert len(results[0].fix_suggestions) > 0
    assert any("algorithmic change" in str(f) for f in results[0].fix_suggestions)

def test_multiple_loops(pred):
    code = """void foo(int N, double *A, double *B) {
        for (int i = 0; i < N; i++) {
            A[i] = i;
        }
        for (int j = 0; j < N; j++) {
            B[j] = A[j] * 2;
        }
    }
    """
    results, _ = pred.predict(code)
    assert len(results) == 2
    assert results[0].loop_line == 2
    assert results[1].loop_line == 5

def test_json_serializable_output(pred):
    code = """
    void add(int N, double *A, double *B, double *C) {
        for (int i = 0; i < N; i++) {
            C[i] = A[i] + B[i];
        }
    }
    """
    results, _ = pred.predict(code)
    assert len(results) == 1
    r = results[0]
    
    # Check if we can convert it to a dict and dump it
    d = {
        "line": r.loop_line,
        "verdict": r.verdict,
        "confidence": r.confidence,
        "crossover_n": r.crossover_n,
        "features": r.features
    }
    
    json_str = json.dumps(d)
    assert "verdict" in json_str
    assert "features" in json_str
    assert "crossover_n" in json_str

    code = """
    void gemm(int N, double *A, double *B, double *C) {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                double sum = 0.0;
                for (int k = 0; k < N; k++) {
                    sum += A[i*N + k] * B[k*N + j];
                }
                C[i*N + j] = sum;
            }
        }
    }
    """
    results, _ = pred.predict(code)
    
    # 1. Print the actual length of results array (proof it filtered the inner loops out)
    print("\n--- GEMM TEST OUTPUT ---")
    print(f"Total loops analyzed: {len(results)}")
    
    # The parser returns all 3 loops, but predictor should filter down to just the outermost one
    assert len(results) == 1
    
    # 2. Print the line number (proof it caught the outer 'for (int i...' loop at line 3)
    print(f"Analyzed Loop Line: {results[0].loop_line}")
    assert results[0].loop_line == 2
    
    # 3. Print the nested depth (proof it knows there are 2 loops inside it)
    print(f"Calculated Nest Depth: {results[0].features['nest_depth']}")
    assert results[0].features['nest_depth'] == 2.0
    
    print(f"Verdict: {results[0].verdict} ({results[0].confidence:.2%})")
    print("------------------------")

def test_nested_loops_gemm(pred):
    code = """
    void gemm(int N, double *A, double *B, double *C) {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                double sum = 0.0;
                for (int k = 0; k < N; k++) {
                    sum += A[i*N + k] * B[k*N + j];
                }
                C[i*N + j] = sum;
            }
        }
    }
    """
    results, _ = pred.predict(code)
    
    print("\n--- GEMM TEST OUTPUT ---")
    print(f"Total loops analyzed: {len(results)}")
    assert len(results) == 1
    
    print(f"Analyzed Loop Line: {results[0].loop_line}")
    assert results[0].loop_line == 2
    
    print(f"Calculated Nest Depth: {results[0].features['nest_depth']}")
    assert results[0].features['nest_depth'] == 2.0
    
    print(f"Verdict: {results[0].verdict} ({results[0].confidence:.2%})")
    
    print("\n--- FEATURES DICT (Sanity Check) ---")
    import json
    print(json.dumps(results[0].features, indent=2))
    print("------------------------")
def test_mismatched_literal_bounds(pred):
    code = """
    void rectangular(double *A) {
        for (int i = 0; i < 100; i++) {
            for (int j = 0; j < 50; j++) {
                for (int k = 0; k < 10; k++) {
                    A[i*500 + j*10 + k] *= 2.0;
                }
            }
        }
    }
    """
    results, _ = pred.predict(code)
    
    print("\n--- RECTANGULAR TEST OUTPUT ---")
    assert len(results) == 1
    
    # 100 * 50 * 10 = 50000
    # math.log2(50000) = 15.60964...
    import math
    expected_log = math.log2(50000)
    actual_log = results[0].features['trip_count_log']
    
    print(f"Expected Trip Count Log: {expected_log}")
    print(f"Actual Trip Count Log: {actual_log}")
    
    # Check that it didn't just cube the outermost (100^3 = 1,000,000 -> log2 = 19.93)
    # or cube the default (1000^3 = 1,000,000,000 -> log2 = 29.89)
    assert abs(actual_log - expected_log) < 1e-5
    print("Mismatched explicit bounds successfully multiplied! (100 * 50 * 10 = 50,000)")
    print("------------------------")