#include <curand_kernel.h>

#include <iostream>
#include <cuda_profiler_api.h>

#define HEADS true
#define TAILS false

#define CUDA(x) { gpuAssert((x), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__global__ void random_to_tosses(const float* source, bool* dest, int N)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > N) return; // only needed if threads*blocks is not a multiple of N // TODO
    
    dest[i] = source[i] >= 0.5;
}

__global__ void find_targets(const bool* tosses, int* dest, int N)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x + 3;
    if (i > N - 3) return; // only needed if threads*blocks is not a multiple of N // TODO

    // >= 3 heads, followed by 3 tails
    if (tosses[i - 3] == HEADS && tosses[i - 2] == HEADS && tosses[i - 1] == HEADS
        && tosses[i] == TAILS && tosses[i + 1] == TAILS && tosses[i + 2] == TAILS) {

        // automatically take the known steps
        dest[i - 3] = -3;
        dest[i + 2] = 2;
    }
}

__device__ bool allThreadsDone;
__global__ void find_dists(bool* is_done, int* old_dist, int* new_dist, int N)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > N) return;

    int dist = old_dist[i];

    if (dist == 0) {
        new_dist[i] = 0;
        return;
    }

    bool propagated = false;
    bool completed = false;

    // if the (right) neighbour is in range
    if (is_done[i]) {
        new_dist[i] = dist;
    } else if (dist > 0 && i < N - 1) {
        // if the 1st/2nd neighbour is populated, then place result in new_dist
        if (old_dist[i + 1] < 0) {
            new_dist[i] = 2 * dist + 1;
            new_dist[i + 1] = 0;
            completed = true;
        } else if (i < N - 2 && old_dist[i + 2] < 0) {
            new_dist[i] = 2 * dist + 2;
            new_dist[i + 2] = 0;
            completed = true;
        } else {
            new_dist[i + 1] = dist + 1;
            new_dist[i] = 0;
            propagated = true;
        }
    // if there are no (left) neighbour in range
    } else if (dist < 0 && i > 1 && old_dist[i - 1] == 0 && i < N - 2 && old_dist[i - 2] == 0) {
        new_dist[i - 1] = dist - 1;
        new_dist[i] = 0;
        propagated = true;
    }

    is_done[i] = is_done[i] || completed;
    allThreadsDone = !(completed || propagated);
}

int main() {
    int N = 1 << 24;

    float* tossSource;
    bool* tosses;
    int* targets;
    int* intermediate;
    bool* tracker;

    // assign memory required
    CUDA(cudaMalloc(&tossSource, N * sizeof(float)));
    CUDA(cudaMalloc(&tosses, N * sizeof(bool)));
    CUDA(cudaMallocManaged(&targets, N * sizeof(int)));
    CUDA(cudaMallocManaged(&intermediate, N * sizeof(int)));
    CUDA(cudaMalloc(&tracker, N * sizeof(bool)));

    bool initialValue = false;
    CUDA(cudaMemcpyToSymbol(allThreadsDone, &initialValue, sizeof(bool)));

    // create the RNG
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, (unsigned long long int) clock());

    // generate random numbers 0-1
    curandGenerateUniform(gen, tossSource, N);

    // convert the floats to booleans representing coin flips
    random_to_tosses<<<N, 1>>>(tossSource, tosses, N);
    CUDA(cudaPeekAtLastError());

    // mark all the locations of the desired >=3 heads, 3 tails pattern with a 1 in an 0-filled array
    CUDA(cudaMemset(targets, 0, N * sizeof(int))); 
    find_targets<<<N, 1>>>(tosses, targets, N);
    CUDA(cudaPeekAtLastError());

    // find the distances between the marked locations
    CUDA(cudaMemset(tracker, false, N * sizeof(bool)));
    bool allThreadsDoneHost;

    do {
        find_dists<<<N, 1 >>> (tracker, targets, intermediate, N);
        CUDA(cudaPeekAtLastError());

        CUDA(cudaMemcpyFromSymbol(&allThreadsDoneHost, allThreadsDone, sizeof(bool)));

        int* swap = targets;
        targets = intermediate;
        intermediate = swap;
    } while (!allThreadsDoneHost);

    // wait for all calculations to stop
    CUDA(cudaDeviceSynchronize());

    // find average
    double average = 0.0f;
    int count = 0;
    for (int i = 0; i < N; i++) {
        if (targets[i] != 0) {
            average += targets[i];
            count++;
        }
    }
    printf("Avg: %f\n", average / count);

    // free all used memory
    CUDA(cudaFree(tossSource));
    CUDA(cudaFree(tosses));
    CUDA(cudaFree(targets));
    CUDA(cudaFree(intermediate));
    CUDA(cudaFree(tracker));

    // flush profiling
    cudaProfilerStop();

    // exit
    return EXIT_SUCCESS;
}
