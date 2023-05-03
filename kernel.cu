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

// TODO add CURAND checking

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

// blocks and threads must be power of two
/*_device__ int filtered;
__global__ void reduce_sparse(const int* input, int* output, int N)
{
    __shared__ int block_data[];

    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N && input[i] != 0) {
        output[atomicAdd(filtered, 1)] = input[i];
    }
}*/

int main() { // TODO add CUDA_CALL/CURAND_CALL error checking macros
    // TODO: Process overlapping chunks i.e. for 1,000,000 'flips', where we want 4 heads, 3 tails
    // TODO: we would need 7 to consider if this condition is met, so for 0-7, 1-8, 2-9, ..., 999 993 - 1 000 000
    // TODO: Check if this condition is met, then return this in another array
    // TODO: Then this just needs to be counted as a 1 or 0, this can probably be combined with the last step
    // TODO: Using a similar approach to the histogram

    int N = 1 << 24;

    float* tossSource;
    bool* tosses;
    int* targets;
    int* intermediate;
    bool* tracker; // todo place in device code
    // int* reduced;

    // assign memory required
    CUDA(cudaMalloc(&tossSource, N * sizeof(float)));
    CUDA(cudaMalloc(&tosses, N * sizeof(bool)));
    CUDA(cudaMallocManaged(&targets, N * sizeof(int)));
    CUDA(cudaMallocManaged(&intermediate, N * sizeof(int)));
    CUDA(cudaMalloc(&tracker, N * sizeof(bool)));
    // CUDA(cudaMallocManaged(&reduced, N * sizeof(int)));

    bool initialValue = false;
    CUDA(cudaMemcpyToSymbol(allThreadsDone, &initialValue, sizeof(bool))); // TODO test in-place assignment

    // create the RNG
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, (unsigned long long int) clock());

    // generate random numbers 0-1
    curandGenerateUniform(gen, tossSource, N);

    // convert the floats to booleans representing coin flips
    random_to_tosses<<<N, 1>>>(tossSource, tosses, N); // TODO optimise grid/block settings
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
    } while (!allThreadsDoneHost); // TODO turn into while loop on GPU

    // reduce targets to a smaller array of distances with 0s eliminated
    //reduce_sparse<<<(N + 255) / 256, 256>>>(targets, reduced, N); // TODO further reduction

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
    // CUDA(cudaFree(reduced));

    // flush profiling
    cudaProfilerStop();

    // exit
    return EXIT_SUCCESS;

    //    int blockSize = 256; // threads per block
    //    int numBlocks = (N + blockSize - 1) / blockSize; // blocks needed to compute all values
    //    add<<<numBlocks, blockSize>>>(N, x, y);
}
