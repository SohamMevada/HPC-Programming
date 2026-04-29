#ifndef INIT_H
#define INIT_H

#include <stdio.h>

// Point structure
typedef struct {
    double x, y;
    bool is_void;
} Points;

// Global simulation parameters
extern int GRID_X, GRID_Y, NX, NY;
extern int NUM_Points_Global, Maxiter;
extern double dx, dy;

// Initialization & I/O
//void read_points(FILE *file, Points *points);
void read_points(FILE *f, Points *p, int n);
void initializepoints(Points *points);

#endif
