import requests
import json
import time
import subprocess
import os

# Start the server
server_proc = subprocess.Popen(["python", "web/server.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=os.getcwd())
time.sleep(2)  # wait for server to start

try:
    tests = {
        "1. Empty input": "",
        "2. No loops": "int add(int a, int b) { return a + b; }",
        "3. Syntax error": "void f(int N) { for (int i = 0 i < N; i++) { A[i] = 1; }",
        "4. Not C at all": "hello this is not code at all just some words.",
        "5. Very large input": "void huge(int *A, int N) {\n" + "".join([f"for(int i=0; i<N; i++) A[i] = {j};\n" for j in range(25)]) + "}",
        "6. Unicode in source": "// 测试 unicode комментарий 🚀\nvoid f(int *A, int N) { for(int i=0; i<N; i++) A[i] = 1; }",
        "7. Zero iterations": "void f(int *A) { for (int i = 0; i < 0; i++) { A[i] = 1; } }"
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
            
            try:
                # Try pretty printing JSON
                print("Response JSON:")
                print(json.dumps(resp.json(), indent=2)[:1000] + ("\n... [TRUNCATED]" if len(json.dumps(resp.json(), indent=2)) > 1000 else ""))
            except:
                # Fallback to text
                print("Response Text:")
                print(resp.text[:1000])
        except Exception as e:
            print(f"Request failed: {e}")
            
finally:
    server_proc.kill()
