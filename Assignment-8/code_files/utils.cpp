#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <mpi.h>
#include <omp.h>
#include "init.h"
#include "utils.h"

double min_val, max_val;

void interpolation(double *mesh_value, Points *points, int n) {
    const int ms = GRID_X * GRID_Y;
    const int nt = omp_get_max_threads();
    const double inv_dx=1/dx;
    const double inv_dy=1/dy;
    const double dx_c=dx;
    const double dy_c=dy;
    const int gx=GRID_X;
    // Allocate a buffer for every thread (Local Reduction Pad)
    double* pad = (double*) calloc(nt * ms, sizeof(double));

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        double* t_mesh = &pad[tid * ms];
        
        // 1. Parallel Particle Processing (No Atomics Needed!)
        #pragma omp for schedule(static)
        for(int p = 0; p < n; p++) {
            if(points[p].is_void) continue;
            const double px = points[p].x;
            const double py = points[p].y;

            const int i = int(px * inv_dx);
            const int j = int(py * inv_dy);

            const double lx = px - (i * dx_c);
            const double ly = py - (j * dy_c);

            const double dx_lx = dx_c - lx;
            const double dy_ly = dy_c - ly;

            // Weights
            const double w00 = dx_lx * dy_ly;
            const double w10 = lx    * dy_ly;
            const double w01 = dx_lx * ly;
            const double w11 = lx    * ly;

            const int base = j * gx + i;

            t_mesh[base]         += w00;
            t_mesh[base + 1]     += w10;
            t_mesh[base + gx]    += w01;
            t_mesh[base + gx + 1]+= w11;
        }

        // 2. Parallel Local Grid Reduction (Sum threads into process-mesh)
        #pragma omp for schedule(static)
        for(int m = 0; m < ms; m++) {
            double sum = 0.0;
            for(int t = 0; t < nt; t++) sum += pad[t * ms + m];
            mesh_value[m] = sum;
        }
    }
    free(pad);
}

void get_global_bounds(double *mesh_value) {
    double l_min = DBL_MAX, l_max = -DBL_MAX;
    #pragma omp parallel for reduction(min:l_min) reduction(max:l_max)
    for (int i = 0; i < GRID_X * GRID_Y; i++) {
        if (mesh_value[i] < l_min) l_min = mesh_value[i];
        if (mesh_value[i] > l_max) l_max = mesh_value[i];
    }
    MPI_Allreduce(&l_min, &min_val, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&l_max, &max_val, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
}

void mover(double *mesh_value, Points *points, int n) {
    double range = max_val - min_val;
    double inv_range = (range > 1e-12) ? (2.0 / range) : 0.0;
    const double inv_dx=1/dx;
    const double inv_dy=1/dy;
    const double dx_c=dx;
    const double dy_c=dy;
    const int gx=GRID_X;
    const double a=2/(max_val-min_val);
    const double b=-(max_val+min_val)/(max_val-min_val);
    #pragma omp parallel for schedule(static)
    for (int p = 0; p < n; p++) {
        if (points[p].is_void) continue;


        double px = points[p].x;
        double py = points[p].y;

        const int i = int(px * inv_dx);
        const int j = int(py * inv_dy);

        const double lx = px - (i * dx_c);
        const double ly = py - (j * dy_c);

        const double dx_lx = dx_c - lx;
        const double dy_ly = dy_c - ly;

        const int base = j * gx + i;

        // Compute Force
        const double F = (dx_lx * dy_ly) * (a*mesh_value[base]+b) +
                         (lx    * dy_ly) * (a*mesh_value[base + 1]+b) +
                         (dx_lx * ly)    * (a*mesh_value[base + gx]+b) +
                         (lx    * ly)    * (a*mesh_value[base + gx + 1]+b);
        
        px += dx_c * F;
        py += dy_c * F;

        // Simplify branching logic
        points[p].is_void = (px <= 0 || px >= 1 || py <= 0 || py >= 1);
        points[p].x = px;
        points[p].y = py;
    }
}

void save_mesh(double *mesh_value) {
    FILE *fd = fopen("Mesh.out", "w");
    for (int i = 0; i < GRID_Y; i++) {
        for (int j = 0; j < GRID_X; j++) fprintf(fd, "%lf ", mesh_value[i * GRID_X + j]);
        fprintf(fd, "\n");
    }
    fclose(fd);
}
long long int void_count(Points *points, int n) {
    long long int count = 0;
    #pragma omp parallel for reduction(+:count)
    for (int i = 0; i < n; i++) {
        count += (points[i].is_void ? 1 : 0);
    }
    return count;
}