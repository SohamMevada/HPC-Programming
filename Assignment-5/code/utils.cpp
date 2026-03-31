#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "utils.h"
#include <random>
#include <chrono>
#include <iostream>
#include <vector>
#include <cstring>
#include <cmath>
#include <fstream>
#include <string>
#define RNG_POOL_SIZE 1024
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

struct RNG {
    uint32_t state;
    inline float next() {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        // return (float)state / (float)4294967295U;
        return (float)state * 2.3283064365386963e-10f;
    }
};

int partitionInPlace(Points* points,int iteration, int N, double dx, double dy) {
    const double tx = iteration * dx;
    const double ty = iteration * dy;
    
    int left = 0;
    int right = N - 1;

    while (left <= right) {
        bool is_bad = (points[left].x < tx || points[left].x > 1.0 - tx ||
                       points[left].y < ty || points[left].y > 1.0 - ty);
        
        if (is_bad) {
            Points temp = points[left];
            points[left] = points[right];
            points[right] = temp;
            if (right == 0) break;
            right--;
        } else {
            left++;
        }
    }
    return left;
}

void mover_parallel_deferred_del(Points* points, int N, int split_idx, double dx, double dy, int** thread_buffers, int* local_bad_counts, int* global_offsets) {
    int BAD_THREADS=(NUM_THREADS>2)?2:1;
    int SAFE_THREADS=NUM_THREADS-BAD_THREADS;
    #pragma omp parallel num_threads(NUM_THREADS)
    {
        int tid = omp_get_thread_num();
        RNG rng = { (uint32_t)(123456789 + tid + (int)(points[0].x * 1000)) };
        if (tid < SAFE_THREADS){
            int chunk = split_idx / SAFE_THREADS;
            int start = tid * chunk;
            int end = (tid == SAFE_THREADS - 1) ? split_idx : (start + chunk);
            for (int i = start; i < end; ++i) {
                points[i].x += (rng.next() - 0.5) * dx;
                points[i].y += (rng.next() - 0.5) * dy;
            }
        }
        else if(BAD_THREADS>=2){
            int bad_tid = tid - SAFE_THREADS;
            int bad_section_size = N - split_idx;
            int chunk = bad_section_size / BAD_THREADS;
            int start = split_idx + (bad_tid * chunk);
            int end = (tid == NUM_THREADS - 1) ? N : split_idx + ((bad_tid + 1) * chunk);

            int* my_buffer = thread_buffers[tid];
            int my_count = 0;

            for (int i = start; i < end; ++i) {
                points[i].x += (rng.next() - 0.5f) * dx;
                points[i].y += (rng.next() - 0.5f) * dy;

                if ((points[i].x < 0.0) | (points[i].x > 1.0) | 
                    (points[i].y < 0.0) | (points[i].y > 1.0)) {
                    
                    points[i].x = (double)rng.next();
                    points[i].y = (double)rng.next();

                    my_buffer[my_count++] = i;
                }
            }
            local_bad_counts[tid] = my_count;
        }
        else{
            size_t current_end = N;
            for (size_t i = split_idx; i < current_end; ++i) {
                points[i].x += (rng.next() - 0.5) * dx;
                points[i].y += (rng.next() - 0.5) * dy;
                bool outx = (points[i].x < 0.0) || (points[i].x > 1.0);
                bool outy = (points[i].y < 0.0) || (points[i].y > 1.0);

                if (outx|outy) {
                    points[i].x=(outx)?rng.next()*dx:points[i].x;
                    points[i].y=(outy)?rng.next()*dy:points[i].y;
                    Points p_temp = points[i];
                    points[i] = points[current_end - 1];
                    points[current_end - 1] = p_temp;

                        current_end--;
                        i--; 
                }
            }
        }
        if(BAD_THREADS>1)
        {
            #pragma omp barrier 

            #pragma omp single
            {
                int current_offset = 0;
                for (int i = NUM_THREADS - 1; i >= SAFE_THREADS; --i) {
                    global_offsets[i] = N - current_offset - local_bad_counts[i];
                    current_offset += local_bad_counts[i];
                }
            }
            if (tid >= SAFE_THREADS) {
                int* my_buffer = thread_buffers[tid];
                int my_count = local_bad_counts[tid];
                int my_target_start = global_offsets[tid];

                for (int j = 0; j < my_count; ++j) {
                    int source_idx = my_buffer[j];
                    int target_idx = my_target_start + j;

                    Points temp = points[source_idx];
                    points[source_idx] = points[target_idx];
                    points[target_idx] = temp;
                }
            }
        }
    }
}


bool add_and_check(uint64_t* s,Points* p,double dx,double dy)
{
    double rnd1=0x1p-64;
    double rnd2=0x1p-64;
    
    *s ^= *s << 13;
    *s ^= *s >> 7;
    rnd1*=*s;
    *s ^= *s << 17;
    rnd2*=*s;
    p->x+=rnd1*dx;
    p->y+=rnd2*dy;
    bool outx=((p->x)*(1-p->x)) > 0.0;
    if(!outx)
        return outx;
    bool outy=((p->y)*(1-p->y)) > 0.0;
    if(!outy)
        return outy;
    
    return true;
}

void swap(Points* p1,Points* p2)
{
    p1->x=p1->x+p2->x;
    p2->x=p1->x-p2->x;
    p1->y=p1->y+p2->y;
    p2->y=p1->y-p2->y;
}


int serial_del(Points* p,double dx,double dy,int start,int end)
{
    bool* valid=nullptr;
    valid=(bool*)calloc(end-start+1,sizeof(bool));    
    int i=start;
    int j=end;
    uint64_t *s=nullptr;
    s=new uint64_t;
    *s=100000;
    while(i<=j)
    {
        if(!valid[i-start])
            valid[i-start]=add_and_check(s,&p[i],dx,dy);
        if(!valid[j-start])
            valid[j-start]=add_and_check(s,&p[j],dx,dy);
        
        if(valid[i-start]&&valid[j-start])
        {
            i++;
        }
        else if(!valid[i-start]&&valid[j-start])
        {
            swap(&p[i],&p[j]);
            valid[i-start]=true;
            valid[j-start]=false;
            j--;
            i++;
        }
        else if(valid[i-start]&&!valid[j-start])
        {
            i++;
            j--;
        }
        else
        {
            j--;
            swap(&p[i],&p[j]);
            valid[i-start]=false;
            valid[j-start]=false;
            j--;
        }
    }
    free(valid);
    
    while(i<NUM_Points)
    {
        double rnd1=0x1p-64;
        double rnd2=0x1p-64;
        *s ^= *s << 13;
        *s ^= *s >> 7;
        rnd1*=*s;
        *s ^= *s << 17;
        rnd2*=*s;
        p[i].x=rnd1;
        p[i].y=rnd2;
        i++;
    }

    return j;
}

int merge_chunks(Points* p, int s1, int b1, int s2, int b2) {
    // Left side "bad" range: [b1 + 1, s2 - 1]
    // Right side "good" range: [s2, b2]
    
    int pl = b1 + 1;         // Start of left-bad
    int pr = s2;             // Start of right-good
    int left_bad_end = s2 - 1;
    int right_good_end = b2;

    // Calculate how many points we can actually swap
    int left_bad_count = left_bad_end - pl + 1;
    int right_good_count = right_good_end - pr + 1;
    
    // We swap the minimum of the two counts
    int swap_count = (left_bad_count < right_good_count) ? left_bad_count : right_good_count;

    // Optimized swap loop
    // Using a pointer-based loop helps the compiler vectorize the move
    Points* src = &p[pl];
    Points* dest = &p[pr];

    for (int i = 0; i < swap_count; ++i) {
        Points temp = src[i];
        src[i] = dest[i];
        dest[i] = temp;
    }

    // The new "end of good points" for the merged block is:
    // Original good points in left (b1) + newly moved good points from right
    return b1 + right_good_count;
}

void mover_serial_deferred_del(Points* p,double dx,double dy)
{
    int b=serial_del(p,dx,dy,0,NUM_Points-1);
}

void mover_serial_immdt_del(Points* points,double dx,double dy){
    RNG rnd;
    rnd.state=1000000;
    for(int i=0;i<NUM_Points;i++)
    {
        points[i].x+=(rnd.next()-0.5)*dx;
        points[i].y+=(rnd.next()-0.5)*dy;
        if(points[i].x<0.0||points[i].x>1.0||points[i].y<0.0||points[i].y>1.0)
        {
            points[i].x=(rnd.next());
            points[i].y=(rnd.next()); 
        }
    }
}

vector<int> ident_bad_points(const Points* points, int num_points, double dx, double dy, int num_iter) {
    double max_dist = sqrt((num_iter * dx * num_iter * dx) + (num_iter * dy * num_iter * dy));
    double min_safe = max_dist;
    double max_safe = 1.0 - max_dist;

    std::vector<int> bad_indices;

    #pragma omp parallel
    {
        std::vector<int> local_bads;
        
        #pragma omp for schedule(static)
        for(int i = 0; i < num_points; i++) {
            double x = points[i].x;
            double y = points[i].y;

            // Flag if the point is too close to the edge
            if (x < min_safe || x > max_safe || y < min_safe || y > max_safe) {
                local_bads.push_back(i);
            }
        }

        // Merge local vectors into the global vector safely
        #pragma omp critical
        {
            bad_indices.insert(bad_indices.end(), local_bads.begin(), local_bads.end());
        }
    }
    return bad_indices;
}

void mover_parallel_immdt_del_fst(Points* points, int num_points, double dx, double dy, const std::vector<int>& bad_indices)
{
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        RNG rnd;
        rnd.state = 10000 + tid;

        double rnd_pool[RNG_POOL_SIZE];
        for(int i = 0; i < RNG_POOL_SIZE; i++) {
            rnd_pool[i] = rnd.next() - 0.5;
        }

        int pool_idx = 0;

        #pragma omp for schedule(static)
        for(int i = 0; i < num_points; i++) {
            double rx = rnd_pool[pool_idx] * dx;
            double ry = rnd_pool[pool_idx + 1] * dy;
            
            // Fast bitwise wrap-around
            pool_idx = (pool_idx + 2) & (RNG_POOL_SIZE - 1); 

            points[i].x += rx;
            points[i].y += ry;
        }
        #pragma omp master
        {
            size_t num_bad = bad_indices.size();
            for(size_t k = 0; k < num_bad; k++) {
                int idx = bad_indices[k];
                points[idx].x -= floor(points[idx].x);
                points[idx].y -= floor(points[idx].y);
            }
        }
    } 
}


void mover_parallel_immdt_del(Points* points, float dx, float dy) {
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        uint32_t s = 123456789 + tid * 100; 

        // Force the loop to vectorize if the compiler supports it
        #pragma omp for schedule(static)
        for (int i = 0; i < NUM_Points; i++) {
            // Xorshift 32
            s ^= s << 13; s ^= s >> 17; s ^= s << 5;
            uint32_t vx = (s >> 9) | 0x3F800000; // In the range 1 to 2
            float rx = *(float*)&vx - 1.5f; 

            s ^= s << 13; s ^= s >> 17; s ^= s << 5;
            uint32_t vy = (s >> 9) | 0x3F800000;
            float ry = *(float*)&vy - 1.5f;

            float new_x = points[i].x + (rx * dx);
            float new_y = points[i].y + (ry * dy);
            if (new_x < 0.0f || new_x >= 1.0f || new_y < 0.0f || new_y >= 1.0f)
            {
                s ^= s << 13; s ^= s >> 17; s ^= s << 5;
                vx = (s >> 9) | 0x3F800000;
                new_x = *(float*)&vx - 1.0f; // Range [0.0, 1.0)

                s ^= s << 13; s ^= s >> 17; s ^= s << 5;
                vy = (s >> 9) | 0x3F800000;
                new_y = *(float*)&vy - 1.0f;
            }

            points[i].x = new_x;
            points[i].y = new_y;
        }
    }
}
struct ThreadRNG {
    uint64_t state;
};

void mover_parallel(Points *points, double deltaX, double deltaY)
{
    int num_threads = omp_get_max_threads(); 
    
    std::vector<ThreadRNG> rngs(num_threads);
    for(int i = 0; i < num_threads; i++) {
        rngs[i].state = 1000 + i; 
    }
    
    const double inv_max = 5.42101086242752217e-20;
    
    #pragma omp parallel num_threads(num_threads)
    {
        int tid = omp_get_thread_num();
        uint64_t s = rngs[tid].state; 
        double rnd;
        
        #pragma omp for schedule(static)
        for (int i = 0; i < NUM_Points; i++)
        {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            
            rnd = s * inv_max;

            points[i].x = (double)((long long)(points[i].x + (rnd * deltaX)));
            points[i].y = (double)((long long)(points[i].y + (rnd * deltaY)));
        }
        rngs[tid].state = s; 
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

// Function to generate the 3D histogram and save to a file
void generate_3d_histogram(const Points* points, int num_points, int grid_size, const std::string& filename) {
    // 1. Create a flattened 2D grid array: grid[grid_size][grid_size]
    // We use a flat 1D vector of size (grid_size * grid_size) for continuous memory access.
    std::vector<int> histogram(grid_size * grid_size, 0);

    // Calculate the width of each bin
    double bin_width = 1.0 / grid_size;

    // 2. Parallel binning of particles
    #pragma omp parallel
    {
        // Thread-local histogram to avoid cache contention and atomic operations
        std::vector<int> local_hist(grid_size * grid_size, 0);

        #pragma omp for schedule(static)
        for (int i = 0; i < num_points; i++) {
            // Find which bin the point belongs to (scale 0.0-1.0 to 0-(grid_size-1))
            int bin_x = (int)(points[i].x / bin_width);
            int bin_y = (int)(points[i].y / bin_width);

            // Guard against edge cases where x or y might exactly equal 1.0
            if (bin_x >= grid_size) bin_x = grid_size - 1;
            if (bin_y >= grid_size) bin_y = grid_size - 1;
            if (bin_x < 0) bin_x = 0;
            if (bin_y < 0) bin_y = 0;

            // Map the 2D grid coordinate to a 1D array index
            int idx = bin_x * grid_size + bin_y;
            local_hist[idx]++;
        }

        // Merge local thread grids into the global grid safely
        #pragma omp critical
        {
            for (int i = 0; i < grid_size * grid_size; i++) {
                histogram[i] += local_hist[i];
            }
        }
    }

    // 3. Write data to a CSV file for plotting
    std::ofstream file(filename);
    file << "x_center,y_center,count\n"; // Header

    for (int i = 0; i < grid_size; i++) {
        for (int j = 0; j < grid_size; j++) {
            // Calculate the geometric center of this bin for accurate plotting
            double x_center = (i * bin_width) + (bin_width / 2.0);
            double y_center = (j * bin_width) + (bin_width / 2.0);
            int count = histogram[i * grid_size + j];

            file << x_center << "," << y_center << "," << count << "\n";
        }
    }
    file.close();
    std::cout << "3D Histogram saved to " << filename << std::endl;
}

// Backup

// void mover_parallel(Points *points, double deltaX, double deltaY)
// {
//     int num_threads = omp_get_max_threads(); // Or omp_get_max_threads()
//     ThreadRNG rngs[num_threads]={1000};
    
//     // Pre-seed RNGs outside the parallel region
//     // for(int i=0; i<num_threads; i++) rngs[i].state = 88172645463325252ULL ^ i;
//     const double inv_max = numeric_limits<double>::min();
//     #pragma omp parallel num_threads(num_threads)
//     {
//         int tid = omp_get_thread_num();
//         uint64_t s = rngs[tid].state; // Local copy of state to avoid cache bouncing
//         double rnd;
//         #pragma omp for schedule(static)
//         for (int i = 0; i < NUM_Points; i++)
//         {
//             // Xorshift algorithm: extremely fast, branchless
//             s ^= s << 13;
//             s ^= s >> 7;
//             s ^= s << 17;
            
//             // Logically get 0 or 1 without a distribution object
//             rnd = s*inv_max;
//             // cout<<rnd<<endl;
//             // Manual "floor" via casting (only if points are positive)
//             // This avoids the overhead of the math.h library call
//             points[i].x = (double)((long long)(points[i].x + (rnd * deltaX)));
//             points[i].y = (double)((long long)(points[i].y + (rnd * deltaY)));
//         }
//         rngs[tid].state = s; // Write back state once
//     }
// }