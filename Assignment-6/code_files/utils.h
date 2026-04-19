#ifndef UTILS_H
#define UTILS_H
#include <time.h>
#include "init.h"
extern int NUM_Threads;
void serial_interpolation(double * __restrict mesh_value,const Points * __restrict points) ;
void parallel_interpolation(double* mesh_value,Points* points,LocalMeshPad& pad);
void save_mesh(double *mesh_value);

#endif