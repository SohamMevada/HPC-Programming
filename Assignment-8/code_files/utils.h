#ifndef UTILS_H
#define UTILS_H
#include "init.h"

extern double min_val, max_val;

void interpolation(double *mesh_value, Points *points, int n);
void get_global_bounds(double *mesh_value);
void mover(double *mesh_value, Points *points, int n);
long long int void_count(Points *points, int n);
void save_mesh(double *mesh_value);
#endif