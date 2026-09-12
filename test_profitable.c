void f(double *p, int N) {
    for (int i = 0; i < 1000000; i++) {
        *p++ = i;
    }
}