#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <stdint.h>
#include "init.h"
#include "utils.h"

// Global variables
int GRID_X, GRID_Y, NX, NY;
int NUM_Points, Maxiter;
int NUM_THREADS;
double dx, dy;
int MAX_LOCAL_BUFFER;

int main(int argc, char **argv) {

    NX = 1000;
    NY = 400;
    NUM_Points = 14000000;
    Maxiter = 10;

    GRID_X = NX + 1;
    GRID_Y = NY + 1;
    dx = 1.0 / NX;
    dy = 1.0 / NY;

    int thread_list[] = {2, 4, 8, 16};
    int num_threads = 4;

    FILE *fp = fopen("speedup_vs_threads_normal_250_100.txt", "w");
    if (fp == NULL) {
        printf("Error opening file!\n");
        return 1;
    }

    fprintf(fp, "Threads\tMover_Time\tSpeedup\n");

    double *mesh_value = (double *) calloc(GRID_X * GRID_Y, sizeof(double));
    Points *points = (Points *) calloc(NUM_Points, sizeof(Points));

    if (!mesh_value || !points) {
        printf("Memory allocation failed!\n");
        return 1;
    }

    initializepoints(points);

    size_t idx = 0;
    double serial_move_time = 0.0;

    for (int iter = 0; iter < Maxiter; iter++) {

        interpolation(mesh_value, points);

        clock_t start_move = clock();
        // mover_serial_deferred_del(points,dx,dy);
        mover_serial(points,dx,dy);
        clock_t end_move = clock();

        double move_time = (double)(end_move - start_move) / CLOCKS_PER_SEC;
        serial_move_time += move_time;
    }

    printf("\nSerial mover total time = %lf\n", serial_move_time);

    free(mesh_value);
    free(points);

    for (int t = 0; t < num_threads; t++) {

        NUM_THREADS = thread_list[t];
        omp_set_num_threads(NUM_THREADS);
        omp_set_dynamic(0);
        printf("\n=== Threads = %d ===\n", NUM_THREADS);

        double *mesh_value = (double *) calloc(GRID_X * GRID_Y, sizeof(double));
        Points *points = (Points *) calloc(NUM_Points, sizeof(Points));

        if (!mesh_value || !points) {
            printf("Memory allocation failed!\n");
            free(mesh_value);
            free(points);
            continue;
        }

        initializepoints(points);

        size_t idx = 0;
        double total_move_time = 0.0;
        int* local_bad_counts = new int[NUM_THREADS];
        int* global_offsets = new int[NUM_THREADS];
        int** thread_buffers = new int*[NUM_THREADS];
        
        for (int iter = 0; iter < Maxiter; iter++) {

            interpolation(mesh_value, points);

            clock_t start_move = clock();

            // if (iter == 0) {
            //     idx = partitionInPlace(points, Maxiter, NUM_Points, dx, dy);
            //     MAX_LOCAL_BUFFER=(NUM_Points-idx);
            //     for(int i=0; i<NUM_THREADS; i++) thread_buffers[i] = new int[MAX_LOCAL_BUFFER];
            // }
            // mover_parallel_deferred_del(points,NUM_Points,idx,dx,dy,thread_buffers,local_bad_counts,global_offsets);
            mover_parallel(points,dx,dy);
            clock_t end_move = clock();

            double move_time = (double)(end_move - start_move) / CLOCKS_PER_SEC;
            total_move_time += move_time;
        }

        double speedup = serial_move_time / total_move_time;

        printf("Mover Time = %lf | Speedup = %lf\n",
               total_move_time, speedup);

        fprintf(fp, "%d\t%lf\t%lf\n",
                NUM_THREADS,
                total_move_time,
                speedup);

        free(mesh_value);
        free(points);
    }

    fclose(fp);

    return 0;
}
// #include <stdio.h>
// #include <stdlib.h>
// #include <time.h>
// #include <omp.h>
// #include <stdint.h>
// #include "init.h"
// #include "utils.h"

// // Global variables (UNCHANGED + added NUM_THREADS)
// int GRID_X, GRID_Y, NX, NY;
// int NUM_Points, Maxiter;
// int NUM_THREADS;
// double dx, dy;

// int main(int argc, char **argv) {

//     NX = 1000;
//     NY = 400;
//     NUM_Points = 14000000;
//     Maxiter = 10;

//     GRID_X = NX + 1;
//     GRID_Y = NY + 1;
//     dx = 1.0 / NX;
//     dy = 1.0 / NY;

//     int thread_list[] = {2, 4, 8, 16};
//     int num_threads = 4;

//     FILE *fp = fopen("scalability_results.txt", "w");
//     if (fp == NULL) {
//         printf("Error opening file!\n");
//         return 1;
//     }

//     fprintf(fp, "Threads\tTotal_Interp\tTotal_Mover\tGrand_Total\n");

//     for (int t = 0; t < num_threads; t++) {

//         NUM_THREADS = thread_list[t];
//         omp_set_num_threads(NUM_THREADS);

//         printf("\n=== Threads = %d ===\n", NUM_THREADS);

//         double *mesh_value = (double *) calloc(GRID_X * GRID_Y, sizeof(double));

//         // ✅ This line is correct IF typedef exists
//         Points *points = (Points *) calloc(NUM_Points, sizeof(Points));

//         if (!mesh_value || !points) {
//             printf("Memory allocation failed!\n");
//             free(mesh_value);
//             free(points);
//             continue;
//         }

//         initializepoints(points);

//         size_t idx = 0;
//         double total_interp_time = 0.0;
//         double total_move_time = 0.0;

//         printf("Iter\tInterp\t\tMover\t\tTotal\n");

//         for (int iter = 0; iter < Maxiter; iter++) {

//             clock_t start_interp = clock();
//             interpolation(mesh_value, points);
//             clock_t end_interp = clock();

//             clock_t start_move = clock();

//             if (iter == 0) {
//                 idx = partitionInPlace(points, NUM_Points, dx, dy);
//             }

//             moverIteration1(points, NUM_Points, idx, dx, dy);

//             clock_t end_move = clock();

//             double interp_time = (double)(end_interp - start_interp) / CLOCKS_PER_SEC;
//             double move_time = (double)(end_move - start_move) / CLOCKS_PER_SEC;

//             total_interp_time += interp_time;
//             total_move_time += move_time;

//             printf("%d\t%lf\t%lf\t%lf\n",
//                    iter + 1,
//                    interp_time,
//                    move_time,
//                    interp_time + move_time);
//         }

//         double grand_total = total_interp_time + total_move_time;

//         printf("TOTAL\t%lf\t%lf\t%lf\n",
//                total_interp_time, total_move_time, grand_total);

//         fprintf(fp, "%d\t%lf\t%lf\t%lf\n",
//                 NUM_THREADS,
//                 total_interp_time,
//                 total_move_time,
//                 grand_total);

//         save_mesh(mesh_value);

//         free(mesh_value);
//         free(points);
//     }

//     fclose(fp);

//     return 0;
// }