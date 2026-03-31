#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>
#include <stdint.h>
#include "init.h"
#include "utils.h"

// Global variables (UNCHANGED)
int GRID_X, GRID_Y, NX, NY;
int NUM_Points, Maxiter;
double dx, dy;
int NUM_THREADS=1;

int main(int argc, char **argv) {

    Maxiter = 10;

    // Grid configurations
    int NX_list[] = {250, 500, 1000};
    int NY_list[] = {100, 200, 400};

    // Particle configurations
    int particle_counts[] = {
        100,
        10000,
        1000000,
        100000000,
        1000000000
    };

    int num_grids = 3;
    int num_particles = 5;

    // File to store results
    FILE *fp = fopen("serial_mover_inplace_insertion.txt", "w");
    if (fp == NULL) {
        printf("Error opening file!\n");
        return 1;
    }

    fprintf(fp, "NX\tNY\tPoints\tTotal_Interp\tTotal_Mover\tGrand_Total\n");

    for (int g = 0; g < num_grids; g++) {

        NX = NX_list[g];
        NY = NY_list[g];

        GRID_X = NX + 1;
        GRID_Y = NY + 1;
        dx = 1.0 / NX;
        dy = 1.0 / NY;

        for (int p = 0; p < num_particles; p++) {

            NUM_Points = particle_counts[p];

            printf("\n=== NX=%d NY=%d Points=%d ===\n", NX, NY, NUM_Points);

            double *mesh_value = (double *) calloc(GRID_X * GRID_Y, sizeof(double));
            Points *points = (Points *) calloc(NUM_Points, sizeof(Points));

            if (!mesh_value || !points) {
                printf("Memory allocation failed!\n");
                free(mesh_value);
                free(points);
                continue;
            }

            initializepoints(points);

            printf("Iter\tInterp\t\tMover\t\tTotal\n");

            size_t idx = 0;

            double total_interp_time = 0.0;
            double total_move_time = 0.0;

            for (int iter = 0; iter < Maxiter; iter++) {

                clock_t start_interp = clock();
                interpolation(mesh_value, points);
                clock_t end_interp = clock();

                clock_t start_move = clock();
                mover_serial_immdt_del(points,dx,dy);
                clock_t end_move = clock();

                double interp_time = (double)(end_interp - start_interp) / CLOCKS_PER_SEC;
                double move_time = (double)(end_move - start_move) / CLOCKS_PER_SEC;
                double total = interp_time + move_time;

                total_interp_time += interp_time;
                total_move_time += move_time;

                printf("%d\t%lf\t%lf\t%lf\n",
                       iter + 1, interp_time, move_time, total);
            }

            double grand_total = total_interp_time + total_move_time;

            printf("TOTAL\t%lf\t%lf\t%lf\n",
                   total_interp_time, total_move_time, grand_total);

            fprintf(fp, "%d\t%d\t%d\t%lf\t%lf\t%lf\n",
                    NX, NY, NUM_Points,
                    total_interp_time,
                    total_move_time,
                    grand_total);

            save_mesh(mesh_value);

            free(mesh_value);
            free(points);
        }
    }

    fclose(fp);

    return 0;
}