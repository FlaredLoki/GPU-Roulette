import requests
import json
import time
import subprocess
import os
import shutil

# Start the server
server_proc = subprocess.Popen(["python", "web/server.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=os.getcwd())
time.sleep(3)

model_path = "models/gpugate_model.pkl"
backup_path = "models/gpugate_model.pkl.bak"
code = "void f(int N) { for(int i=0;i<N;i++); }"

try:
    print(f"1. Copying {model_path} to backup.")
    shutil.copy2(model_path, backup_path)
    
    print(f"2. Overwriting {model_path} with garbage bytes.")
    with open(model_path, 'w') as f:
        f.write("not a pickle")
        
    print("3. Hitting /api/predict on the live server.")
    resp = requests.post("http://127.0.0.1:5000/api/predict", json={"code": code})
    print(f"Status Code: {resp.status_code}")
    print(f"Response: {resp.text.strip()[:500]}")
    
    print("4. Restoring backup.")
    shutil.move(backup_path, model_path)
    
    print("5. Hitting /api/predict again.")
    resp2 = requests.post("http://127.0.0.1:5000/api/predict", json={"code": code})
    print(f"Status Code: {resp2.status_code}")
    print(f"Response: {resp2.text.strip()[:500]}")
    
finally:
    server_proc.kill()
    if os.path.exists(backup_path):
        shutil.move(backup_path, model_path)
