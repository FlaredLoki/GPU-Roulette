import requests
import json
import time
import subprocess
import os
import concurrent.futures

# Start the server
server_proc = subprocess.Popen(["python", "web/server.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=os.getcwd())
time.sleep(3)

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

def make_request(req_id):
    start = time.time()
    try:
        resp = requests.post("http://127.0.0.1:5000/api/predict", json={"code": gemm_code})
        elapsed = time.time() - start
        if resp.status_code == 200:
            d = resp.json()["loops"][0]
            return req_id, resp.status_code, d["verdict"], d["confidence"], elapsed
        else:
            return req_id, resp.status_code, resp.text, None, elapsed
    except Exception as e:
        elapsed = time.time() - start
        return req_id, "ERROR", str(e), None, elapsed

try:
    print(f"\n{'='*60}\n===== Concurrent Determinism Test =====\n{'='*60}")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(make_request, i) for i in range(1, 4)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
    for r in sorted(results, key=lambda x: x[0]):
        req_id, status, verdict, conf, elapsed = r
        print(f"Request {req_id}: HTTP {status} | Verdict: {verdict} | Confidence: {conf} | Time: {elapsed:.3f}s")

finally:
    # Get server output
    server_proc.terminate()
    try:
        outs, errs = server_proc.communicate(timeout=2)
        print("\n--- Server Stderr (Logs) ---")
        print(errs.decode('utf-8', errors='ignore'))
    except:
        pass
