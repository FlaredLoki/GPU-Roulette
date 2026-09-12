from src.predictor import GPUGatePredictor
import json

code_a = 'void f(double *a) { for (int i = 0; i < 1000; i++) { if (a[i] < 0) break; a[i] *= 2; } }'
code_b = 'void f(double *a) { for (int i = 0; i < 1000; i++) { if (a[i] < 0) continue; a[i] *= 2; } }'
code_c = 'void f(double *a) { for (int i = 0; i < 1000; i++) { if (a[i] < 0) goto done; a[i] *= 2; } done:; }'

pred = GPUGatePredictor()

def analyze(name, code):
    results, errors = pred.predict(code)
    for r in results:
        out = {
            "name": name,
            "loops": [
                {
                    "verdict": r.verdict,
                    "confidence": r.confidence,
                    "trip_count": r.feature_vector.loop_info.trip_count,
                    "trip_count_log": r.features["trip_count_log"]
                }
            ]
        }
        print(json.dumps(out, indent=2))

analyze('a) break', code_a)
analyze('b) continue', code_b)
analyze('c) goto', code_c)
