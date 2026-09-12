import requests
import json
import time
import subprocess
import os

# Start the server
server_proc = subprocess.Popen(["python", "web/server.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=os.getcwd())
time.sleep(5)  # wait for server to start

try:
    tests = {
        "1. Empty input": ""
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
