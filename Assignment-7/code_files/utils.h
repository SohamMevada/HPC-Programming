#ifndef UTILS_H
#define UTILS_H
#include <time.h>
#include "init.h"
extern double min_val, max_val;
typedef struct {
    // A single contiguous block: [NUM_Threads * GRID_X * GRID_Y]
    double* local_meshes; 
    int mesh_size; // GRID_X * GRID_Y
} LocalMeshPad;
// PIC operations
void interpolation(double *mesh_value, Points *points,LocalMeshPad& pad);
void normalization(double *mesh_value);
void mover(double *mesh_value, Points *points);
void denormalization(double *mesh_value);
long long int void_count(Points *points);
void save_mesh(double *mesh_value);

#endif
