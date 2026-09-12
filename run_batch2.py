import subprocess

tests = {
    "Test1_LoopCarriedDep": """
void dep_test(double *A, double *B, int N) {
    for (int i = 1; i < 1000000; i++) {
        A[i] = A[i-1] * 2.0 + B[i];
    }
}
""",
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
    "Test3_OpaqueCall": """
double my_custom_transform(double x);
void opaque_test(double *A, int N) {
    for (int i = 0; i < 1000000; i++) {
        A[i] = my_custom_transform(A[i]);
    }
}
""",
    "Test4_PointerAlias": """
void f(double *A, double *B, int N) {
    for (int i = 0; i < 1000000; i++) {
        A[i] = B[i] * 2.0;
    }
}
""",
    "Test5_IndirectReduction": """
void indirect_reduce(double *A, int *idx, int N) {
    double sum = 0.0;
    for (int i = 0; i < 1000000; i++) {
        sum += A[idx[i]];
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
