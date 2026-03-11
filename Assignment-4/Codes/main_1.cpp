#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>

#include "init.h"
#include "utils.h"

// Global variables
int GRID_X, GRID_Y, NX, NY;
int NUM_Points, Maxiter;
double dx, dy;

int main(int argc, char **argv) {

    Maxiter = 10;

    int NX_vals[] = {250, 500, 1000};
    int NY_vals[] = {100, 200, 400};
    long long Point_vals[] = {
        100LL,
        10000LL,
        1000000LL,
        100000000LL,
        1000000000LL
    };

    int num_grid_configs = 3;
    int num_point_configs = 1;

    printf("NX\tNY\tPoints\t\tInterp_Time\n");
    printf("------------------------------------------------------\n");
    NUM_Points = Point_vals[3];

            // Allocate memory
            

            Points *points =
                (Points *) calloc((size_t)NUM_Points, sizeof(Points));


            // double *mesh_value =
            //     (double *) calloc((size_t)GRID_X * GRID_Y, sizeof(double));

            // if (mesh_value == NULL || points == NULL) {
            //     printf("Memory allocation failed for NX=%d NY=%d Points=%d\n",
            //             NX, NY, NUM_Points);
            //     exit(1);
            // }

            initializepoints(points);


    for (int g = 0; g < num_grid_configs; g++) {

        NX = NX_vals[g];
        NY = NY_vals[g];

        GRID_X = NX + 1;
        GRID_Y = NY + 1;
        dx = 1.0 / NX;
        dy = 1.0 / NY;
            


            double *mesh_value =
                (double *) calloc((size_t)GRID_X * GRID_Y, sizeof(double));
                
            if (mesh_value == NULL || points == NULL) {
                printf("Memory allocation failed for NX=%d NY=%d Points=%d\n",
                        NX, NY, NUM_Points);
                exit(1);
            }
        
            double total_interp_time = 0.0;

            for (int iter = 0; iter < Maxiter; iter++) {

                clock_t start_interp = clock();
                interpolation(mesh_value, points);
                clock_t end_interp = clock();

                double interp_time =
                    (double)(end_interp - start_interp) / CLOCKS_PER_SEC;

                total_interp_time += interp_time;
            }

            // total_interp_time ;

            printf("%d\t%d\t%d\t\t%lf\n",
                    NX, NY, NUM_Points, total_interp_time);

            free(mesh_value);
        
    }

            free(points);

    return 0;
}