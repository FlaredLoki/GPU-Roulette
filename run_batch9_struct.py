from src.predictor import GPUGatePredictor
import json

code_struct = """
struct Point { float x, y; };
void f() {
    struct Point arr[1000];
    for (int i = 0; i < 1000; i++) {
        arr[i].x = arr[i].x * 2.0f;
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
                    "mem_reads": f.get("mem_reads"),
                    "mem_writes": f.get("mem_writes"),
                    "stride_one_ratio": f.get("stride_one_ratio"),
                    "unique_arrays": f.get("unique_arrays"),
                    "verdict": r.verdict,
                    "confidence": r.confidence
                }
            ]
        }
        print(json.dumps(out, indent=2))

analyze('Struct Array Access', code_struct)
