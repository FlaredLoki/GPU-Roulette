from src.predictor import GPUGatePredictor
import json

code_p2p = """
void f1(double **A, double **B, double **C, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = A[i][j] + B[i][j];
        }
    }
}
"""

code_flat = """
void f2(double A[100][100], double B[100][100], double C[100][100], int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = A[i][j] + B[i][j];
        }
    }
}
"""

pred = GPUGatePredictor()

def analyze(name, code):
    results = pred.predict(code)
    for r in results:
        f = r.features
        out = {
            "name": name,
            "loops": [
                {
                    "data_reuse_score": f["data_reuse_score"],
                    "stride_one_ratio": f["stride_one_ratio"],
                    "verdict": r.verdict,
                    "confidence": r.confidence
                }
            ]
        }
        print(json.dumps(out, indent=2))

analyze('1) double **A (Pointer-to-Pointer)', code_p2p)
analyze('2) double A[100][100] (Flat Array)', code_flat)
