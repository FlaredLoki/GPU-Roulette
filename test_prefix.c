void prefix(int N, double *A) { for (int i = 1; i < N; i++) { A[i] = A[i] + A[i-1]; } }
