#include <math.h>
#include "utils.h"
#include <cstdlib>

// Problem 01
void matrix_multiplication_ijk(double** m1, double** m2, double** result, int N) {
    for(int i=0;i<N;i++){
        for(int j=0;j<N;j++){
            for(int k=0;k<N;k++){
                result[i][j]+=m1[i][k]*m2[k][j];
            }
        }    
    }

}

void matrix_multiplication_jik(double** m1, double** m2, double** result, int N) {
    for(int j=0;j<N;j++){
        for(int i=0;i<N;i++){
            for(int k=0;k<N;k++){
                result[i][j]+=m1[i][k]*m2[k][j];
            }
        }    
    }

}

void matrix_multiplication_ikj(double** m1, double** m2, double** result, int N) {
    for(int i=0;i<N;i++){
        for(int k=0;k<N;k++){
            for(int j=0;j<N;j++){
                result[i][j]+=m1[i][k]*m2[k][j];
            }
        }    
    }

}

void matrix_multiplication_kji(double** m1, double** m2, double** result, int N) {
    for(int k=0;k<N;k++){
        for(int j=0;j<N;j++){
            for(int i=0;i<N;i++){
                result[i][j]+=m1[i][k]*m2[k][j];
            }
        }    
    }

}

void matrix_multiplication_jki(double** m1, double** m2, double** result, int N) {
    for(int j=0;j<N;j++){
        for(int k=0;k<N;k++){
            for(int i=0;i<N;i++){
                result[i][j]+=m1[i][k]*m2[k][j];
            }
        }    
    }

}

void matrix_multiplication_kij(double** m1, double** m2, double** result, int N) {
    for(int k=0;k<N;k++){
        for(int i=0;i<N;i++){
            for(int j=0;j<N;j++){
                result[i][j]+=m1[i][k]*m2[k][j];
            }
        }    
    }

}

void transpose(double** m, double** mt, int N) {
    for(int i=0;i<N;i++){
        for(int j=0;j<N;j++){
           mt[i][j]=m[j][i];
        }
    }
}

void transposed_matrix_multiplication(double** m1, double** m2, double** result, int N) {
    double **m2t = (double**)malloc(N * sizeof(double*));

    for (int i = 0; i < N; i++) {
        m2t[i] = (double*)malloc(N * sizeof(double));   
    }

    for(int i=0;i<N;i++)
    {
        for(int j=0;j<N;j++)
        {
            m2t[i][j]=m2[j][i];
        }
    }

    for(int i=0;i<N;i++){
        for(int j=0;j<N;j++){
            for(int k=0;k<N;k++){
                result[i][j]+=m1[i][k]*m2t[j][k];
            }
        }    
    }
}

void block_matrix_multiplication(double** m1, double** m2, double** result, int B, int N) {
    int Nb=N/B;
    
    for(int a=0;a<N;a+=B){
        for(int b=0;b<N;b+=B){
            for(int d=0;d<N;d+=B)
            {
                // matrix_multiplication_1(&m1[a][b],&m2[b][d],&result[a][d],B,N);
                for(int i=a;i<a+B;i++){
                    for(int k=b;k<b+B;k++){
                        for(int j=d;j<d+B;j++){
                            result[i][j]+=m1[i][k]*m2[k][j];
                        }
                    }    
                }
            }
        }    
    }
}