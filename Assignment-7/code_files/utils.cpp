#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "utils.h"
#include <omp.h>
#include <limits>

double min_val = std::numeric_limits<double>::max();
double max_val = std::numeric_limits<double>::lowest();

extern int NUM_Threads;

// interpolation 
void interpolation(double* mesh_value, Points* points, LocalMeshPad& pad) {
    const int np = NUM_Points;
    const int nt = NUM_Threads;
    const int gx = GRID_X;
    const int gy = GRID_Y;
    const int ms = gx * gy;
    const double inv_dx = 1.0 / dx;
    const double inv_dy = 1.0 / dy;
    const double dx_c = dx;
    const double dy_c = dy;

    #pragma omp parallel num_threads(nt)
    {
        int tid = omp_get_thread_num();
        double* t_mesh = &pad.local_meshes[tid * ms];
        
        // 1. Parallel Particle Processing
        #pragma omp for schedule(static)
        for(int p = 0; p < np; p++) {
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

        // 2. PARALLEL Reduction (This fixes your scaling issue)
        // Instead of 1 thread summing everything, all threads sum a chunk of the grid.
        #pragma omp for schedule(static)
        for(int m = 0; m < ms; m++) {
            double sum = 0.0;
            for(int t = 0; t < nt; t++) {
                sum += pad.local_meshes[t * ms + m];
            }
            mesh_value[m] += sum;
        }
    }
}

void normalization(double *mesh_value) {
    min_val = std::numeric_limits<double>::max();
    max_val = std::numeric_limits<double>::lowest();
    const int ms = GRID_X * GRID_Y;

    // 1. Let OpenMP handle the reduction inherently (Faster and prevents false sharing)
    #pragma omp parallel for reduction(min:min_val) reduction(max:max_val) num_threads(NUM_Threads)
    for(int m = 0; m < ms; m++) {
        if(mesh_value[m] < min_val) min_val = mesh_value[m];
        if(mesh_value[m] > max_val) max_val = mesh_value[m];
    }

    const double diff = max_val - min_val;
    // Prevent divide-by-zero if mesh is perfectly uniform
    const double a = (diff != 0.0) ? (2.0 / diff) : 1.0; 
    const double b = (diff != 0.0) ? (-(max_val + min_val) / diff) : 0.0;

    // 2. Parallel Normalization
    #pragma omp parallel for num_threads(NUM_Threads)
    for(int m = 0; m < ms; m++) {
        mesh_value[m] = a * mesh_value[m] + b;
    }
}

// mover via reverse-interpolation
void mover(double *mesh_value, Points *points) {
    const int np = NUM_Points;
    const int gx = GRID_X;
    const double inv_dx = 1.0 / dx;
    const double inv_dy = 1.0 / dy;
    const double dx_c = dx;
    const double dy_c = dy;

    #pragma omp parallel for schedule(static) num_threads(NUM_Threads)
    for(int p = 0; p < np; p++) {
        if(points[p].is_void) continue;

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
        const double F = (dx_lx * dy_ly) * mesh_value[base] +
                         (lx    * dy_ly) * mesh_value[base + 1] +
                         (dx_lx * ly)    * mesh_value[base + gx] +
                         (lx    * ly)    * mesh_value[base + gx + 1];
        
        px += dx_c * F;
        py += dy_c * F;

        // Simplify branching logic
        points[p].is_void = (px <= 0 || px >= 1 || py <= 0 || py >= 1);
        points[p].x = px;
        points[p].y = py;
    }
}

void denormalization(double *mesh_value) {
    const double a = (max_val - min_val) * 0.5;
    const double b = (max_val + min_val) * 0.5;
    const int ms = GRID_X * GRID_Y;

    #pragma omp parallel for schedule(static) num_threads(NUM_Threads)
    for(int i = 0; i < ms; i++) {
        mesh_value[i] = a * mesh_value[i] + b;
    }
}

// count particles that went beyond the domain
long long int void_count(Points *points) {
    long long int voids = 0;
    
    // You missed the parallelization here entirely!
    #pragma omp parallel for reduction(+:voids) num_threads(NUM_Threads)
    for (int i = 0; i < NUM_Points; i++) {
        if (points[i].is_void) {
            voids++;
        }
    }
    return voids;
}

// Write mesh to file
void save_mesh(double *mesh_value) {
    FILE *fd = fopen("Mesh.out", "w");
    if (!fd) {
        printf("Error creating Mesh.out\n");
        exit(1);
    }

    for (int i = 0; i < GRID_Y; i++) {
        for (int j = 0; j < GRID_X; j++) {
            fprintf(fd, "%lf ", mesh_value[i * GRID_X + j]);
        }
        fprintf(fd, "\n");
    }

    fclose(fd);
}