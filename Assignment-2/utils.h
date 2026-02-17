#ifndef UTILS_H
#define UTILS_H
#include <time.h>

void matrix_multiplication_ijk(double** m1, double** m2, double** result, int N);
void matrix_multiplication_jik(double** m1, double** m2, double** result, int N);
void matrix_multiplication_ikj(double** m1, double** m2, double** result, int N);
void matrix_multiplication_kji(double** m1, double** m2, double** result, int N);
void matrix_multiplication_jki(double** m1, double** m2, double** result, int N);
void matrix_multiplication_kij(double** m1, double** m2, double** result, int N);
void transpose(double** m, double** mt, int N);
void transposed_matrix_multiplication(double** m1, double** m2, double** result, int N);
void block_matrix_multiplication(double** m1, double** m2, double** result, int B, int N);

#endif
