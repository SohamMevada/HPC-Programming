#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "utils.h"
#include <random>
#include <chrono>
#include <iostream>
using namespace std;
// Interpolation (Serial Code)
void interpolation(double *mesh_value, Points *points) {
    double inv_dx = (double)NX; 
    double inv_dy = (double)NY;

    for (int p = 0; p < NUM_Points; p++) {
        double x = points[p].x;
        double y = points[p].y;

        int i = (int)(x * inv_dx);
        int j = (int)(y * inv_dy);

        // Ternary operators compile to 'cmov' (Conditional Move) -> No branch penalty
        i = (i >= NX) ? NX - 1 : i;
        j = (j >= NY) ? NY - 1 : j;

        double lx = x - (i * dx);
        double ly = y - (j * dy);

        double wx_m = dx - lx;
        double wy_m = dy - ly;

        int base_idx = j * GRID_X + i;

        mesh_value[base_idx]              += wx_m * wy_m; 
        mesh_value[base_idx + 1]          += lx * wy_m; 
        mesh_value[base_idx + GRID_X]     += wx_m * ly; 
        mesh_value[base_idx + GRID_X + 1] += lx * ly; 
    }
}

// Stochastic Mover (Serial Code) 
void mover_serial(Points *points, double deltaX, double deltaY)
{
    uint64_t timeSeed =
        std::chrono::high_resolution_clock::now().time_since_epoch().count();

    std::mt19937 rng((unsigned)timeSeed);

    uint32_t randBits = 0;
    int bitPos = 32;

    for (int i = 0; i < NUM_Points; i++)
    {
        if (bitPos >= 32) {
            randBits = rng();
            bitPos = 0;
        }

        double dx_step = (randBits & (1u << bitPos++)) ? deltaX : -deltaX;

        if (bitPos >= 32) {
            randBits = rng();
            bitPos = 0;
        }

        double dy_step = (randBits & (1u << bitPos++)) ? deltaY : -deltaY;

        double x = points[i].x + dx_step;
        double y = points[i].y + dy_step;

        x += (x < 0.0) * deltaX;
        x -= (x > 1.0) * deltaX;

        y += (y < 0.0) * deltaY;
        y -= (y > 1.0) * deltaY;

        points[i].x = x;
        points[i].y = y;
    }
}

// 64-bit Xorshift state
struct ThreadRNG {
    uint64_t state;
};

void mover_parallel(Points *points, double deltaX, double deltaY)
{
    int num_threads = omp_get_max_threads(); // Or omp_get_max_threads()
    ThreadRNG rngs[num_threads]={88172645463325252ULL};
    
    // Pre-seed RNGs outside the parallel region
    // for(int i=0; i<num_threads; i++) rngs[i].state = 88172645463325252ULL ^ i;
    const double inv_max = numeric_limits<double>::min();
    #pragma omp parallel num_threads(num_threads)
    {
        int tid = omp_get_thread_num();
        uint64_t s = rngs[tid].state; // Local copy of state to avoid cache bouncing
        double rnd;
        #pragma omp for schedule(static)
        for (int i = 0; i < NUM_Points; i++)
        {
            // Xorshift algorithm: extremely fast, branchless
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            
            // Logically get 0 or 1 without a distribution object
            rnd = s*inv_max;
            // cout<<rnd<<endl;
            // Manual "floor" via casting (only if points are positive)
            // This avoids the overhead of the math.h library call
            points[i].x = (double)((long long)(points[i].x + (rnd * deltaX)));
            points[i].y = (double)((long long)(points[i].y + (rnd * deltaY)));
        }
        rngs[tid].state = s; // Write back state once
    }
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