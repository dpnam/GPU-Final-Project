#include <stdint.h>
#include <stdio.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

#define CHECK(call)                                                            \
  {                                                                            \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess) {                                                \
      fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                   \
      fprintf(stderr, "code: %d, reason: %s\n", error,                         \
              cudaGetErrorString(error));                                      \
      exit(1);                                                                 \
    }                                                                          \
  }

struct GpuTimer {
  cudaEvent_t start;
  cudaEvent_t stop;

  GpuTimer() {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
  }

  ~GpuTimer() {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }

  void Start() {
    cudaEventRecord(start, 0);
    cudaEventSynchronize(start);
  }

  void Stop() { cudaEventRecord(stop, 0); }

  float Elapsed() {
    float elapsed;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    return elapsed;
  }
};

void printArray(const uint32_t *a, int n) {
  for (int i = 0; i < n; i++)
    printf("%2i ", a[i]);
  printf("\n");
}

void sortByThrust(const uint32_t *in, int n, uint32_t *out, int nBits) {
  thrust::device_vector<uint32_t> dv_out(in, in + n);
  thrust::sort(dv_out.begin(), dv_out.end());
  thrust::copy(dv_out.begin(), dv_out.end(), out);
}

__global__ void isSorted(const uint32_t *in, int n, bool *sorted) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= n - 1) {
    return;
  }

  if (in[i] > in[i + 1]) {
    *sorted = false;
  }
}

__global__ void computeHistKernel(const uint32_t *in, int n, uint32_t *hist,
                                  int nBins, int bit) {
  // TODO
  extern __shared__ int s_hist[];
  const int i = blockIdx.x * blockDim.x + threadIdx.x;

  for (int s_i = threadIdx.x; s_i < nBins; s_i += blockDim.x) {
    s_hist[s_i] = 0;
  }
  __syncthreads();

  // Each block computes its local hist using atomic on SMEM
  if (i < n) {
    int bin = (in[i] >> bit) & (nBins - 1);
    atomicAdd(&s_hist[bin], 1);
  }

  __syncthreads();

  // transpose
  for (int s_i = threadIdx.x; s_i < nBins; s_i += blockDim.x) {
    hist[gridDim.x * s_i + blockIdx.x] = s_hist[s_i];
  }
}

__global__ void scanBlkKernel(uint32_t *in, int n, uint32_t *out,
                              uint32_t *blkSums) {
  // TODO
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }

  extern __shared__ uint32_t s_in[];
  s_in[threadIdx.x] = in[i];
  __syncthreads();

  for (int stride = 1; stride < blockDim.x; stride *= 2) {
    int strideVal;
    if (threadIdx.x >= stride) {
      strideVal = s_in[threadIdx.x - stride];
    }
    __syncthreads();

    if (threadIdx.x >= stride) {
      s_in[threadIdx.x] += strideVal;
    }
    __syncthreads();
  }

  if (blkSums && threadIdx.x == blockDim.x - 1) {
    blkSums[blockIdx.x] = s_in[threadIdx.x];
  }

  out[i] = s_in[threadIdx.x];
}

// TODO: You can define necessary functions here
__global__ void addBlkSums(uint32_t *in, int n, uint32_t *blkSums) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }

  in[i] += blkSums[blockIdx.x];
}

__global__ void inBlockSort(uint32_t *in, int n, int nBins, int bit,
                            int nBits) {
  
  const int i = blockIdx.x * blockDim.x + threadIdx.x;

  uint32_t s_n = blockDim.x;

  //// init smem
  extern __shared__ uint32_t s_data[];

  // __shared__ uint32_t s_in[256];
  // __shared__ uint32_t s_inBin[256];
  // __shared__ uint32_t s_inBinScan[256];
  // __shared__ uint32_t s_out[256];
  // __shared__ uint32_t s_outBin[256];
  uint32_t *s_in = (uint32_t *)s_data;
  uint32_t *s_inBin = (uint32_t *)(s_in + s_n);
  uint32_t *s_inBinScan = (uint32_t *)(s_inBin + s_n);
  uint32_t *s_out = (uint32_t *)(s_inBinScan + s_n);
  uint32_t *s_outBin = (uint32_t *)(s_out + s_n);

  if (i >= n) {
    s_inBin[threadIdx.x] = nBins - 1;
  } else {
    s_in[threadIdx.x] = in[i];
    s_inBin[threadIdx.x] = (s_in[threadIdx.x] >> bit) & (nBins - 1);
  }
  __syncthreads();

  //// sort smem using radix sort with 1-bit
  for (int b = 0; b < nBits; ++b) {
    /* printf("Doing bit: #%d\n", b); */

    // exclusive scan

    uint32_t temp1, temp2;
    int tid = threadIdx.x;
    temp1 = tid > 0 ? (s_inBin[tid - 1] >> b) & 1 : 0;
    for (int d = 1; d < 32; d <<= 1) {
      temp2 = __shfl_up_sync(0xffffffff, temp1, d);
      if (tid % 32 >= d)
        temp1 += temp2;
    }
    if (tid % 32 == 31)
      s_inBinScan[tid / 32] = temp1;

    __syncthreads();

    if (tid < 32) {
      temp2 = 0;
      if (tid < blockDim.x / 32)
        temp2 = s_inBinScan[tid];
      for (int d = 1; d < 32; d <<= 1) {
        uint32_t temp3 = __shfl_up_sync(0xffffffff, temp2, d);
        if (tid % 32 >= d)
          temp2 += temp3;
      }
      if (tid < blockDim.x / 32)
        s_inBinScan[tid] = temp2;
    }

    __syncthreads();

    if (tid >= 32)
      temp1 += s_inBinScan[tid / 32 - 1];

    __syncthreads();

    s_inBinScan[tid] = temp1;

    __syncthreads();

    // if (threadIdx.x == 0) {
    //   s_inBinScan[threadIdx.x] = 0;
    // } else {
    //   s_inBinScan[threadIdx.x] = ((s_inBin[threadIdx.x - 1] >> b) & 1);
    // }
    // __syncthreads();

    // for (int stride = 1; stride < blockDim.x; stride *= 2) {
    //   int strideVal;
    //   if (threadIdx.x >= stride) {
    //     strideVal = s_inBinScan[threadIdx.x - stride];
    //   }
    //   __syncthreads();

    //   if (threadIdx.x >= stride) {
    //     s_inBinScan[threadIdx.x] += strideVal;
    //   }
    //   __syncthreads();
    // }

    /*
    int thid = threadIdx.x;
    int offset = 1;

    s_inBinScan[2 * thid] = ((s_inBin[2 * thid] >> b) & 1);
    s_inBinScan[2 * thid + 1] = ((s_inBin[2 * thid + 1] >> b) & 1);

    for(int d = blockDim.x >> 1; d > 0; d >>= 1){
      __syncthreads();

      if (thid < d){
        int ai = offset * (2 * thid + 1) - 1;
        int bi = offset * (2 * thid + 2) - 1;

        s_inBinScan[bi] += s_inBinScan[ai];
      }
      offset *= 2;
    }

    if (thid == 0){
      s_inBinScan[blockDim.x - 1] = 0;
    }

    for(int d = 1; d < blockDim.x; d *= 2){

      offset >>= 1;
      __syncthreads();

      if (thid < d){
        int ai = offset * (2 * thid + 1) - 1;
        int bi = offset * (2 * thid + 2) - 1;

        uint32_t tmp = s_inBinScan[ai];
        s_inBinScan[ai] = s_inBinScan[bi];
        s_inBinScan[bi] += tmp;
      }
    }

    __syncthreads();
    */




    

    //scatter
    uint32_t rank;
    if ((s_inBin[threadIdx.x] >> b) & 1) {
      const uint32_t nZeros =
          s_n - s_inBinScan[s_n - 1] - ((s_inBin[s_n - 1] >> b) & 1);
      rank = nZeros + s_inBinScan[threadIdx.x];
    } else {
      rank = threadIdx.x - s_inBinScan[threadIdx.x];
    }
    s_outBin[rank] = s_inBin[threadIdx.x];
    s_out[rank] = s_in[threadIdx.x];

    __syncthreads();

    s_inBin[threadIdx.x] = s_outBin[threadIdx.x];
    s_in[threadIdx.x] = s_out[threadIdx.x];

    __syncthreads();

  }

  in[i] = s_in[threadIdx.x];
}

__global__ void scatter(const uint32_t *in, int n, const uint32_t *histScan,
                        uint32_t *out, int nBins, int bit, int nBits) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;

  uint32_t s_n = blockDim.x;

  //// init smem
  extern __shared__ uint32_t s_data[];

  uint32_t *s_in = (uint32_t *)s_data;
  uint32_t *s_inBin = (uint32_t *)(s_in + s_n);
  uint32_t *s_startIdx = (uint32_t *)(s_inBin + s_n);

  if (i >= n) {
    s_inBin[threadIdx.x] = nBins - 1;
  } else {
    s_in[threadIdx.x] = in[i];
    s_inBin[threadIdx.x] = (s_in[threadIdx.x] >> bit) & (nBins - 1);
  }

  __syncthreads();

  /* out[i] = s_inBin[threadIdx.x]; */

  uint32_t bin = s_inBin[threadIdx.x];
  //// calculate start index
  if (threadIdx.x == 0 || bin != s_inBin[threadIdx.x - 1]) {
    s_startIdx[bin] = threadIdx.x;
  }

  if (i >= n) {
    return;
  }

  __syncthreads();
  //// calculate number of elements at lower index that equals to current
  //// elemement
  uint32_t preCount = threadIdx.x - s_startIdx[bin];

  //// scatter
  uint32_t rank = histScan[gridDim.x * bin + blockIdx.x] + preCount;
  out[rank] = s_in[threadIdx.x];
}

void printDeviceArray(const uint32_t *d_arr, int n) {
  const int BYTES = n * sizeof(*d_arr);
  uint32_t *arr = (uint32_t *)malloc(BYTES);
  cudaMemcpy(arr, d_arr, BYTES, cudaMemcpyDeviceToHost);

  printArray(arr, n);

  free(arr);
}

void checkCorrectness(uint32_t *out, uint32_t *correctOut, int n) {
  for (int i = 0; i < n; i++) {
    if (out[i] != correctOut[i]) {
      printf("INCORRECT :( %d/%d\n", i, n);
      return;
    }
  }
  printf("CORRECT :)\n");
}

// (Partially) Parallel radix sort: implement parallel histogram and parallel
// scan in counting sort Assume: nBits (k in slides) in {1, 2, 4, 8, 16} Why
// "int * blockSizes"? Because we may want different block sizes for diffrent
// kernels:
//   blockSizes[0] for the histogram kernel
//   blockSizes[1] for the scan kernel
void sortByDevice(const uint32_t *in, int n, uint32_t *out, int nBits,
                  int *blockSizes) {
  // TODO
  const int nBins = 1 << nBits;

  const dim3 histBlockSize = dim3(blockSizes[0]);
  const int histBlockCount = (n - 1) / histBlockSize.x + 1;
  const dim3 histGridSize = dim3(histBlockCount);

  const dim3 scanBlockSize = dim3(blockSizes[1]);
  const int scanBlockCount = (nBins * histBlockCount - 1) / scanBlockSize.x + 1;
  const dim3 scanGridSize = dim3(scanBlockCount);

  const size_t ARRAY_BYTES = n * sizeof(uint32_t);
  const size_t HIST_SMEM_BYTES = nBins * sizeof(uint32_t);
  const size_t HIST_BYTES = histBlockCount * HIST_SMEM_BYTES;
  const size_t BLKSUMS_BYTES = scanBlockCount * sizeof(uint32_t);
  const size_t SCAN_SMEM_BYTES = scanBlockSize.x * sizeof(uint32_t);

  const dim3 sortBlockSize = dim3(blockSizes[0]);
  const dim3 sortGridSize = dim3((n - 1) / sortBlockSize.x + 1);

  const size_t SORT_SMEM_BYTES = 2 * 9 * sortBlockSize.x * sizeof(uint32_t);
  const size_t SCATTER_SMEM_BYTES =
      5 * histBlockSize.x * sizeof(uint32_t) + HIST_SMEM_BYTES;

  uint32_t *d_in;
  uint32_t *d_out;
  uint32_t *d_hist; // contains all the transposed local histogram of all blocks
  uint32_t *d_histScan;
  uint32_t *d_blkSums;
  bool *d_sorted;

  uint32_t *blkSums = (uint32_t *)malloc(BLKSUMS_BYTES);

  CHECK(cudaMalloc(&d_in, ARRAY_BYTES));
  CHECK(cudaMalloc(&d_out, ARRAY_BYTES));
  CHECK(cudaMalloc(&d_hist, HIST_BYTES));
  CHECK(cudaMalloc(&d_histScan, HIST_BYTES));
  CHECK(cudaMalloc(&d_blkSums, BLKSUMS_BYTES));
  CHECK(cudaMalloc(&d_sorted, sizeof(bool)));

  CHECK(cudaMemcpy(d_in, in, ARRAY_BYTES, cudaMemcpyHostToDevice));
  CHECK(cudaMemset(d_sorted, 1, sizeof(bool)));

  GpuTimer timer;
  /* printf("IN: "); */
  /* printArray(in, n); */

  for (int bit = 0; bit < 8 * sizeof(uint32_t); bit += nBits) {
    printf("#%d (interation):\n", bit/nBits + 1);

    // Step 1: Calculate local histogram of each block, transpose and copy to d_hist
    timer.Start();
    computeHistKernel<<<histGridSize, histBlockSize, HIST_SMEM_BYTES>>>(
        d_in, n, d_hist, nBins, bit);
    CHECK(cudaGetLastError());
    timer.Stop();
    printf(" + Step 1: Local histogram. Time: %.3f ms\n", timer.Elapsed());

    // Step 2: Scan d_hist
    // scan per block
    timer.Start();
    CHECK(cudaMemset(d_histScan, 0, sizeof(uint32_t)));
    scanBlkKernel<<<scanGridSize, scanBlockSize, SCAN_SMEM_BYTES>>>(
        d_hist, histBlockCount * nBins - 1, d_histScan + 1, d_blkSums);
    CHECK(cudaGetLastError());

    // scan blksums:
    CHECK(
        cudaMemcpy(blkSums, d_blkSums, BLKSUMS_BYTES, cudaMemcpyDeviceToHost));
    for (int i = 1; i < scanBlockCount; ++i) {
      blkSums[i] += blkSums[i - 1];
    }
    CHECK(
        cudaMemcpy(d_blkSums, blkSums, BLKSUMS_BYTES, cudaMemcpyHostToDevice));

    // add scanned blkSums
    addBlkSums<<<scanGridSize, scanBlockSize>>>(
        d_histScan + scanBlockSize.x + 1,
        histBlockCount * nBins - scanBlockSize.x - 1, d_blkSums);
    CHECK(cudaGetLastError());
    timer.Stop();
    printf(" + Step 2: Exclusive scan. Time: %.3f ms\n", timer.Elapsed());

    // Step 3: Scatter
    //// in-block sort
    timer.Start();
    inBlockSort<<<sortGridSize, sortBlockSize, SORT_SMEM_BYTES>>>(
        d_in, n, nBins, bit, nBits);
    CHECK(cudaGetLastError());
    timer.Stop();
    printf(" + Step 3a: In-block sort. Time: %.3f ms\n", timer.Elapsed());

    //// scatter
    timer.Start();
    scatter<<<histGridSize, histBlockSize, SCATTER_SMEM_BYTES>>>(
        d_in, n, d_histScan, d_out, nBins, bit, nBits);
    CHECK(cudaGetLastError());
    timer.Stop();
    printf(" + Step 3b: Scatter. Time: %.3f ms\n", timer.Elapsed());

    uint32_t *tmp = d_in;
    d_in = d_out;
    d_out = tmp;

  }

  cudaMemcpy(out, d_in, ARRAY_BYTES, cudaMemcpyDeviceToHost);

  free(blkSums);

  CHECK(cudaFree(d_in));
  CHECK(cudaFree(d_out));
  CHECK(cudaFree(d_hist));
  CHECK(cudaFree(d_histScan));
  CHECK(cudaFree(d_blkSums));
}

// Radix sort
void sort(const uint32_t *in, int n, uint32_t *out, int nBits,
          bool useThrust = false, int *blockSizes = NULL) {
  GpuTimer timer;
  timer.Start();

  if (useThrust == false) {
    printf("\nRadix sort by thrust\n");
    sortByThrust(in, n, out, nBits);
  } else // use device
  {
    printf("\nRadix sort by device\n");
    sortByDevice(in, n, out, nBits, blockSizes);
  }

  timer.Stop();
  printf("Time: %.3f ms\n", timer.Elapsed());
}

void printDeviceInfo() {
  cudaDeviceProp devProv;
  CHECK(cudaGetDeviceProperties(&devProv, 0));
  printf("**********GPU info**********\n");
  printf("Name: %s\n", devProv.name);
  printf("Compute capability: %d.%d\n", devProv.major, devProv.minor);
  printf("Num SMs: %d\n", devProv.multiProcessorCount);
  printf("Max num threads per SM: %d\n", devProv.maxThreadsPerMultiProcessor);
  printf("Max num warps per SM: %d\n",
         devProv.maxThreadsPerMultiProcessor / devProv.warpSize);
  printf("GMEM: %zu byte\n", devProv.totalGlobalMem);
  printf("SMEM per SM: %zu byte\n", devProv.sharedMemPerMultiprocessor);
  printf("SMEM per block: %zu byte\n", devProv.sharedMemPerBlock);
  printf("****************************\n");
}

int main(int argc, char **argv) {
  // PRINT OUT DEVICE INFO
  printDeviceInfo();

  // SET UP INPUT SIZE
  int n = (1 << 24) + 1;
  /* n = 17; */
  printf("\nInput size: %d\n", n);

  // ALLOCATE MEMORIES
  size_t bytes = n * sizeof(uint32_t);
  uint32_t *in = (uint32_t *)malloc(bytes);
  uint32_t *out = (uint32_t *)malloc(bytes);        // Device result
  uint32_t *correctOut = (uint32_t *)malloc(bytes); // Thrust result

  // SET UP INPUT DATA
  for (int i = 0; i < n; i++)
    in[i] = rand();
  /* in[i] = rand() % 16; */

  // SET UP NBITS
  int nBits = 4; // Default
  if (argc > 1)
    nBits = atoi(argv[1]);
  printf("\nNum bits per digit: %d\n", nBits);

  // DETERMINE BLOCK SIZES
  int blockSizes[2] = {512, 512}; // One for histogram, one for scan
  if (argc == 4) {
    blockSizes[0] = atoi(argv[2]);
    blockSizes[1] = atoi(argv[3]);
  }
  printf("\nHist block size: %d, scan block size: %d\n", blockSizes[0],
         blockSizes[1]);

  // SORT BY THRUST
  sort(in, n, correctOut, nBits);

  // SORT BY DEVICE
  sort(in, n, out, nBits, true, blockSizes);
  checkCorrectness(out, correctOut, n);

  // FREE MEMORIES
  free(in);
  free(out);
  free(correctOut);

  return EXIT_SUCCESS;
}
