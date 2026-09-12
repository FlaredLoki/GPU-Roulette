import requests
import json
import time
import subprocess
import os

# Start the server
server_proc = subprocess.Popen(["python", "web/server.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=os.getcwd())
time.sleep(3)  # wait for server to start

try:
    tests = {
        "3. Syntax error": "void f(int N) { for (int i = 0 i < N; i++) { A[i] = 1; }"
    }

    for name, code in tests.items():
        print(f"\n{'='*60}")
        print(f"===== {name} =====")
        print(f"{'='*60}")
        
        start = time.time()
        try:
            resp = requests.post("http://127.0.0.1:5000/api/predict", json={"code": code})
            print(f"Status Code: {resp.status_code}")
            print(f"Response Time: {(time.time() - start)*1000:.1f}ms")
            print("Response JSON:")
            print(json.dumps(resp.json(), indent=2)[:1000] + ("\n... [TRUNCATED]" if len(json.dumps(resp.json(), indent=2)) > 1000 else ""))
        except Exception as e:
            print(f"Request failed: {e}")
            
finally:
    server_proc.kill()
