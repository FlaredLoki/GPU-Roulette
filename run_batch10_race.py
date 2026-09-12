import requests
import json
import time
import subprocess
import os
import concurrent.futures

gemm_valid = """
void gemm(double *A, double *B, double *C, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i*N+j] = A[i*N+j] + B[i*N+j];
        }
    }
}
"""

gemm_malformed = """
void gemm(double *A, double *B, double *C, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0 j < N; j++) { // MISSING SEMICOLON
            C[i*N+j] = A[i*N+j] + B[i*N+j];
        }
    }
"""

def make_request(req_id, code):
    try:
        resp = requests.post("http://127.0.0.1:5000/api/predict", json={"code": code})
        if resp.status_code == 200:
            d = resp.json()
            return req_id, resp.status_code, d.get("warnings", [])
        else:
            return req_id, resp.status_code, resp.text
    except Exception as e:
        return req_id, "ERROR", str(e)

print(f"\n{'='*60}\n===== Race Condition Test =====\n{'='*60}")

# We will fire 10 valid and 10 malformed interleaved, to heavily stress the server and ensure NO cross-contamination
payloads = []
for i in range(10):
    payloads.append((i*2, gemm_valid))
    payloads.append((i*2+1, gemm_malformed))

with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    futures = [executor.submit(make_request, req_id, code) for req_id, code in payloads]
    results = [f.result() for f in concurrent.futures.as_completed(futures)]
    
for r in sorted(results, key=lambda x: x[0]):
    req_id, status, warnings = r
    expected = "valid" if req_id % 2 == 0 else "malformed"
    actual = "malformed" if len(warnings) > 0 else "valid"
    match = "PASS" if expected == actual else "FAIL"
    print(f"Request {req_id} (Expected {expected}): warnings={len(warnings)} -> {match}")
