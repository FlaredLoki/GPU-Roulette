import subprocess

tests = {
    "Test2_MultiReduction": """
void multi_reduce(double *A, double *B, int N) {
    double sum = 0.0;
    double prod = 1.0;
    for (int i = 0; i < 1000000; i++) {
        sum += A[i];
        prod *= B[i];
    }
}
""",
    "Test4_UncertainPointer": """
void f(double *p, int N) {
    for (int i = 0; i < 1000000; i++) {
        *p++ = i;
    }
}
"""
}

test_file = "test_profitable.c"

for name, code in tests.items():
    with open(test_file, 'w') as f:
        f.write(code.strip())
    
    print(f"\n{'='*60}")
    print(f"===== {name} =====")
    print(f"{'='*60}")
    result = subprocess.run(["python", "scratch_features.py"], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr[:500])
