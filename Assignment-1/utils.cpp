#include <math.h>
#include "utils.h"

void vector_triad_operation(double *x, double *y, double *v, double *S, int Np) {

    for (int p = 0; p < Np; p++) {
        S[p] = x[p] + v[p] * y[p];

        // Prevent compiler from optimizing away the loop
        if (((double)p) == 333.333)
            dummy(p);

    }
}
void copy_operation(double *x, double *y,int Np) {

    for (int p = 0; p < Np; p++) {
        x[p] = y[p];

        // Prevent compiler from optimizing away the loop
        if (((double)p) == 333.333)
            dummy(p);

    }
}
void multiply_operation(double *x, double *y,int Np) {
    double a=5;
    for (int p = 0; p < Np; p++) {
        x[p] = a * y[p];

        // Prevent compiler from optimizing away the loop
        if (((double)p) == 333.333)
            dummy(p);

    }
}
void add_operation(double *x, double *y,int Np, double *s) {
     
    for (int p = 0; p < Np; p++) {
        s[p] = x[p] + y[p];
        // Prevent compiler from optimizing away the loop
        if (((double)p) == 333.333)
            dummy(p);

    }
}

void energy_operation(double *v, double *s,int Np) {
     double m=2;
    for (int p = 0; p < Np; p++) {
        s[p] = 0.5 *m*pow(v[p],2);
        // Prevent compiler from optimizing away the loop
        if (((double)p) == 333.333)
            dummy(p);

    }
}

void dummy(int x) {
    x = 10 * sin(x / 10.0);
}
