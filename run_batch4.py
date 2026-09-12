import requests
import json
import os
import time

def test_endpoint(name, code, expected_status=200):
    print(f"\n{'='*60}\n===== {name} =====\n{'='*60}")
    try:
        resp = requests.post("http://127.0.0.1:5000/api/predict", json={"code": code})
        print(f"Status Code: {resp.status_code}")
        if resp.status_code == expected_status or resp.status_code == 200:
            try:
                print("Response JSON:")
                print(json.dumps(resp.json(), indent=2)[:1500])
            except:
                print("Response Text:", resp.text)
        else:
            print("Response Text:", resp.text)
    except Exception as e:
        print("Request failed:", e)

# Test 1: do-while
do_while_code = """
void f(double *A, int N) {
    int i = 0;
    do {
        A[i] = A[i] * 2.0;
        i++;
    } while (i < N);
}
"""
test_endpoint("1. do-while loop", do_while_code)

# Test 2: Non-I/O blockers
test_endpoint("2a. malloc blocker", "void f(int N) { for (int i=0;i<N;i++) { double *p = malloc(8); p[0] = i; } }")
test_endpoint("2b. free blocker", "void f(double *A, int N) { for (int i=0;i<N;i++) { if(i==N-1) free(A); } }")
test_endpoint("2c. exit blocker", "void f(int N) { for (int i=0;i<N;i++) { if (i<0) exit(1); } }")

# Test 3: Missing model file
print(f"\n{'='*60}\n===== 3. Missing model file =====\n{'='*60}")
os.rename("models/gpugate_model.pkl", "models/gpugate_model_tmp.pkl")
try:
    resp = requests.post("http://127.0.0.1:5000/api/predict", json={"code": "void f(int N) { for(int i=0;i<N;i++); }"})
    print(f"Status Code after rename: {resp.status_code}")
    print(f"Response: {resp.text.strip()}")
except Exception as e:
    print("Request failed:", e)

os.rename("models/gpugate_model_tmp.pkl", "models/gpugate_model.pkl")
try:
    resp = requests.post("http://127.0.0.1:5000/api/predict", json={"code": "void f(int N) { for(int i=0;i<N;i++); }"})
    print(f"Status Code after restore: {resp.status_code}")
except Exception as e:
    print("Request failed:", e)

# Test 4: Determinism
gemm_code = """
void gemm(double *A, double *B, double *C, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}
"""
print(f"\n{'='*60}\n===== 4. Determinism =====\n{'='*60}")
runs = []
try:
    for i in range(3):
        resp = requests.post("http://127.0.0.1:5000/api/predict", json={"code": gemm_code})
        d = resp.json()["loops"][0]
        runs.append({
            "verdict": d["verdict"],
            "confidence": d["confidence"],
            "shap_values": d["grouped_shap"]
        })
        print(f"Run {i+1}: {d['verdict']} ({d['confidence']})")
    
    if runs[0] == runs[1] == runs[2]:
        print("\nAll 3 runs are bit-for-bit identical.")
    else:
        print("\nRuns differ!")
        print(json.dumps(runs, indent=2))
except Exception as e:
    print("Request failed:", e)
