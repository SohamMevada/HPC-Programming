#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <omp.h>
#include "init.h"
#include "utils.h"

// Variables announced as extern in init.h
int GRID_X, GRID_Y, NX, NY, NUM_Points_Global, Maxiter;
double dx, dy;

int main(int argc, char **argv) {
    int rank, size;
    int provided;

    // 1. MPI Initialization
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 3) {
        if (rank == 0) printf("Usage: %s <input_file> <num_threads>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    omp_set_num_threads(atoi(argv[2]));

    // 2. Metadata Handling
    if (rank == 0) {
        FILE *f_meta = fopen(argv[1], "rb");
        if (!f_meta) { MPI_Abort(MPI_COMM_WORLD, 1); }
        fread(&NX, sizeof(int), 1, f_meta);
        fread(&NY, sizeof(int), 1, f_meta);
        fread(&NUM_Points_Global, sizeof(int), 1, f_meta);
        fread(&Maxiter, sizeof(int), 1, f_meta);
        fclose(f_meta);
    }

    MPI_Bcast(&NX, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&NY, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&NUM_Points_Global, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&Maxiter, 1, MPI_INT, 0, MPI_COMM_WORLD);

    GRID_X = NX + 1; GRID_Y = NY + 1;
    dx = 1.0 / NX; dy = 1.0 / NY;
    size_t mesh_elements = (size_t)GRID_X * GRID_Y;

    // 3. Partitioning
    int my_n = NUM_Points_Global / size;
    int start_idx = rank * my_n;
    if (rank == size - 1) my_n += (NUM_Points_Global % size);

    // 4. One-Time Point Loading & Initialization
    Points *local_points = (Points *)malloc(my_n * sizeof(Points));
    FILE *f = fopen(argv[1], "rb");
    if (!f) { MPI_Abort(MPI_COMM_WORLD, 1); }

    // Position file pointer after the 4 metadata ints
    // Note: This assumes input.bin stores (x,y) as doubles or floats. 
    // If input.bin format is strictly (x,y) pairs, we skip the rest of the file.
    long header_offset = 4 * sizeof(int);
    long rank_byte_offset = (long)start_idx * 2 * sizeof(double); // 2 doubles per point (x,y)
    fseek(f, header_offset + rank_byte_offset, SEEK_SET);

    // READ & INITIALIZE (Crucial part for the boolean/velocity)
    for (int i = 0; i < my_n; i++) {
        fread(&local_points[i].x, sizeof(double), 1, f);
        fread(&local_points[i].y, sizeof(double), 1, f);
        local_points[i].is_void = false; // Your boolean update
    }
    fclose(f);

    // 5. Mesh Allocation
    double *global_mesh = (double *)malloc(mesh_elements * sizeof(double));
    double *local_mesh = (double *)malloc(mesh_elements * sizeof(double));

    double t_comp = 0, t_comm = 0;
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    // 6. Simulation Loop (Points are updated in memory by mover)
    for (int iter = 0; iter < Maxiter; iter++) {
        
        // Memset local_mesh to zero so we don't accumulate old data
        memset(local_mesh, 0, mesh_elements * sizeof(double));

        // Computation: Interpolation
        double t1 = omp_get_wtime();
        interpolation(local_mesh, local_points, my_n);
        double t2 = omp_get_wtime();
        t_comp += (t2 - t1);

        // Communication: Allreduce (Implicitly blocks/syncs)
        t1 = MPI_Wtime();
        MPI_Allreduce(local_mesh, global_mesh, (int)mesh_elements, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        get_global_bounds(global_mesh); // Process reduction result
        t2 = MPI_Wtime();
        t_comm += (t2 - t1);

        // Computation: Mover (Updates points in place for the next iteration)
        t1 = omp_get_wtime();
        mover(global_mesh, local_points, my_n);
        t2 = omp_get_wtime();
        t_comp += (t2 - t1);
    }

    double end_time = MPI_Wtime();

    if (rank == 0) {
        save_mesh(global_mesh);
        printf("\n--- Performance Stats ---\n");
        printf("Wall Time: %f s\n", end_time - start_time);
        printf("Compute:   %f s\n", t_comp);
        printf("Comm:      %f s\n", t_comm);
    }

    free(local_points); free(local_mesh); free(global_mesh);
    MPI_Finalize();
    return 0;
}
