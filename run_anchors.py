import subprocess
import os

anchors = {
    "GEMM_N2048": """
void gemm(double A[2048][2048], double B[2048][2048], double C[2048][2048]) {
    for (int i = 0; i < 2048; i++) {
        for (int j = 0; j < 2048; j++) {
            double sum = 0.0;
            for (int k = 0; k < 2048; k++) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
}
""",
    "GEMM_N32": """
void gemm_small(double A[32][32], double B[32][32], double C[32][32]) {
    for (int i = 0; i < 32; i++) {
        for (int j = 0; j < 32; j++) {
            double sum = 0.0;
            for (int k = 0; k < 32; k++) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
}
""",
    "VectorAdd_1M": """
void vector_add(double *A, double *B, double *C, int N) {
    // assume N=1000000
    for (int i = 0; i < 1000000; i++) {
        C[i] = A[i] + B[i];
    }
}
""",
    "Jacobi2D_1024": """
void jacobi_2d(double A[1024][1024], double B[1024][1024]) {
    for (int i = 1; i < 1023; i++) {
        for (int j = 1; j < 1023; j++) {
            B[i][j] = 0.25 * (A[i-1][j] + A[i+1][j] + A[i][j-1] + A[i][j+1]);
        }
    }
}
""",
    "Histogram_scatter": """
void histogram(int *data, int *bins, int N) {
    for (int i = 0; i < 1000000; i++) {
        bins[data[i]]++;
    }
}
""",
    "PrefixScan_naive": """
void prefix_scan(double *A, double *B, int N) {
    for (int i = 1; i < 10000; i++) {
        B[i] = B[i-1] + A[i];
    }
}
""",
    "SYRK_N1024": """
void syrk(double A[1024][1024], double C[1024][1024]) {
    for (int i = 0; i < 1024; i++) {
        for (int j = 0; j <= i; j++) {
            double sum = C[i][j];
            for (int k = 0; k < 1024; k++) {
                sum += A[i][k] * A[j][k];
            }
            C[i][j] = sum;
        }
    }
}
""",
    "BranchHeavyFilter": """
void filter(double *A, double *B, int N) {
    for (int i = 0; i < 1000000; i++) {
        if (A[i] > 0) {
            B[i] = A[i] * 2.0;
        } else if (A[i] < -10) {
            B[i] = A[i] / 2.0;
        } else {
            B[i] = 0;
        }
    }
}
""",
    "PointerChase_LinkedList": """
struct Node { int val; struct Node *next; };
void traverse(struct Node *head) {
    for (struct Node *curr = head; curr != 0; curr = curr->next) {
        curr->val *= 2;
    }
}
""",
    "TinyLoop_16cubed": """
void tiny_loop(double A[16][16][16]) {
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 16; j++) {
            for (int k = 0; k < 16; k++) {
                A[i][j][k] *= 2.0;
            }
        }
    }
}
""",
    "Conv2D_3x3_N1024": """
void conv2d(double in[1024][1024], double out[1024][1024], double kernel[3][3]) {
    for (int i = 1; i < 1023; i++) {
        for (int j = 1; j < 1023; j++) {
            double sum = 0.0;
            for (int ki = -1; ki <= 1; ki++) {
                for (int kj = -1; kj <= 1; kj++) {
                    sum += in[i+ki][j+kj] * kernel[ki+1][kj+1];
                }
            }
            out[i][j] = sum;
        }
    }
}
""",
    "NBody_N16384": """
void nbody(double px[16384], double py[16384], double pz[16384], double fx[16384], double fy[16384], double fz[16384]) {
    for (int i = 0; i < 16384; i++) {
        double fxi = 0.0, fyi = 0.0, fzi = 0.0;
        for (int j = 0; j < 16384; j++) {
            double dx = px[j] - px[i];
            double dy = py[j] - py[i];
            double dz = pz[j] - pz[i];
            double distSqr = dx*dx + dy*dy + dz*dz + 1e-9;
            double invDist = 1.0 / sqrt(distSqr);
            double invDist3 = invDist * invDist * invDist;
            fxi += dx * invDist3;
            fyi += dy * invDist3;
            fzi += dz * invDist3;
        }
        fx[i] = fxi;
        fy[i] = fyi;
        fz[i] = fzi;
    }
}
""",
    "LBM_FluidSim": """
void lbm(double f0[1024][1024], double f1[1024][1024], double f2[1024][1024], double f3[1024][1024]) {
    for (int i = 1; i < 1023; i++) {
        for (int j = 1; j < 1023; j++) {
            // simplified LBM collision and streaming step
            double rho = f0[i][j] + f1[i-1][j] + f2[i][j-1] + f3[i+1][j];
            double u = (f1[i-1][j] - f3[i+1][j]) / rho;
            f0[i][j] = rho * (1.0 - 1.5 * u * u);
            f1[i][j] = rho * (0.5 + u);
        }
    }
}
"""
}

test_file = "test_profitable.c"

for name, code in anchors.items():
    with open(test_file, 'w') as f:
        f.write(code.strip())
    
    print(f"\n===== {name} =====")
    result = subprocess.run(["python", "scratch_features.py"], capture_output=True, text=True)
    
    verdict_line = "ERROR"
    for line in result.stdout.splitlines():
        if "UNPROFITABLE" in line or "PROFITABLE" in line:
            if "=== VERDICT ===" not in line and "remark:" not in line:
                verdict_line = line.strip()
                break
    print(f"Result: {verdict_line}")
