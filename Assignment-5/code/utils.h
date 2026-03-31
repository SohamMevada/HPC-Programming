#ifndef UTILS_H
#define UTILS_H
#include <time.h>
#include "init.h"
#include <stdint.h>
#include <vector>
#include <string>
using namespace std;
extern int NUM_THREADS;
// Previous mover and interpolation functions
void interpolation(double *mesh_value, Points *points);
void mover_serial(Points *points, double deltaX, double deltaY);
void mover_parallel(Points *points, double deltaX, double deltaY);
// Deferred Insertion mover serial and parallel functions
void mover_serial_deferred_del(Points* p,double dx,double dy);
void mover_parallel_deferred_del(Points* points, int N, int split_idx, double dx, double dy, int** thread_buffers, int* local_bad_counts, int* global_offsets);
int partitionInPlace(Points* points,int iteration,int N, double dx, double dy);
// Instantaneous Insertion mover serial and parallel functions
void mover_serial_immdt_del(Points* points,double dx,double dy);
void mover_parallel_immdt_del(Points* points,float dx, float dy);
// Instantaneous Insertion : Extra faster but not immediate
void mover_parallel_immdt_del_fst(Points* points, int num_points, double dx, double dy, const vector<int>& bad_indices);
vector<int> ident_bad_points(const Points* points, int num_points, double dx, double dy, int num_iter);

void save_mesh(double *mesh_value);
void generate_3d_histogram(const Points* points, int num_points, int grid_size, const string& filename);
#endif