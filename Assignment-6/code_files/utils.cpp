#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstring>
#include <stdint.h>

#include "utils.h"

#include <vector>
#include <algorithm>
#include <cmath>

#include <omp.h>


void serial_interpolation(double * __restrict mesh_value,const Points * __restrict points) 
{
    const double inv_dx = 1.0 / dx;
    const double inv_dy = 1.0 / dy;

    for (int p = 0; p < NUM_Points; ++p) {

        const double xi = points[p].x;
        const double yi = points[p].y;

        // Fast index computation (no division)
        const int i = (int)(xi * inv_dx);
        const int j = (int)(yi * inv_dy);

        const double Xi = i * dx;
        const double Yj = j * dy;

        const double lx = xi - Xi;
        const double ly = yi - Yj;

        const double dx_lx = dx - lx;
        const double dy_ly = dy - ly;

        // Weights
        const double w00 = dx_lx * dy_ly;
        const double w10 = lx     * dy_ly;
        const double w01 = dx_lx * ly;
        const double w11 = lx     * ly;

        // Base index (row-major)
        const int base = j * GRID_X + i;

        // Direct accumulation (minimize index math)
        mesh_value[base]                 += w00;
        mesh_value[base + 1]             += w10;
        mesh_value[base + GRID_X]        += w01;
        mesh_value[base + GRID_X + 1]    += w11;
    }
}


void parallel_interpolation(double* mesh_value,Points* points,LocalMeshPad& pad)
{
    const int np=NUM_Points;
    const int nt=NUM_Threads;
    const int gx=GRID_X;
    const int gy=GRID_Y;
    const int ms=gx*gy;
    const int nx=NX;
    const int ny=NY;
    const double inv_dx=NX;
    const double inv_dy=NY;
    const double dx_c=dx;
    const double dy_c=dy;
    #pragma omp parallel num_threads(nt)
    {
        int tid=omp_get_thread_num();
        double* t_mesh=&pad.local_meshes[tid*ms];
        #pragma omp for schedule(static)
        for(int p=0;p<np;p++)
        {
            const double px=points[p].x;
            const double py=points[p].y;

            const int i=int(px*inv_dx);
            const int j=int(py*inv_dy);

            const double Xi = i * dx_c;
            const double Yj = j * dy_c;

            const double lx = px - Xi;
            const double ly = py - Yj;

            const double dx_lx = dx_c - lx;
            const double dy_ly = dy_c - ly;

            // Weights
            const double w00 = dx_lx * dy_ly;
            const double w10 = lx * dy_ly;
            const double w01 = dx_lx * ly;
            const double w11 = lx     * ly;

            // Base index (row-major)
            const int base = j * gx+ i;

            // Direct accumulation (minimize index math)
            t_mesh[base]+= w00;
            t_mesh[base + 1]+= w10;
            t_mesh[base + gx]+= w01;
            t_mesh[base + gx + 1]+= w11;

        }

        // for(int m=0;m<ms;m++)
        // {
        //     #pragma omp atomic
        //     mesh_value[m]+=t_mesh[m];
        // }

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