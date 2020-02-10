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

// Sequential radix sort (paper)
// scan in counting sort Assume: nBits (k in slides) in {1, 2, 4, 8, 16} Why
// "int * blockSizes"? Because we may want different block sizes for diffrent
// kernels:
//   blockSizes[0] for the histogram kernel
//   blockSizes[1] for the scan kernel
void sortByHost(const uint32_t *in, int n, uint32_t *out, int nBits,
                int *blockSizes) {
  // TODO
  const int nBins = 1 << nBits;

  const dim3 histBlockSize = dim3(blockSizes[0]);
  const int histBlockCount = (n - 1) / histBlockSize.x + 1;
  const dim3 histGridSize = dim3(histBlockCount);

  // Not use
  // const dim3 scanBlockSize = dim3(blockSizes[1]);
  // const int scanBlockCount = (nBins * histBlockCount - 1) / scanBlockSize.x +
  // 1; const dim3 scanGridSize = dim3(scanBlockCount);

  uint32_t *hist =
      (uint32_t *)malloc(nBins * histGridSize.x * sizeof(uint32_t));
  uint32_t *histScan =
      (uint32_t *)malloc(nBins * histGridSize.x * sizeof(uint32_t));

  uint32_t *src = (uint32_t *)malloc(n * sizeof(uint32_t));
  memcpy(src, in, n * sizeof(uint32_t));
  uint32_t *originalSrc = src; // Use originalSrc to free memory later
  uint32_t *dst = out;

  /* GpuTimer timer; */
  for (int bit = 0; bit < 8 * sizeof(uint32_t); bit += nBits) {
    /* printf("#%d (iteration):\n", bit / nBits + 1); */

    // Step 1: Calculate local histogram of each block
    /* printf(" + Step 1. Local histogram. "); */
    /* timer.Start(); */

    memset(hist, 0, nBins * histGridSize.x * sizeof(uint32_t));
    for (int blockIndex = 0; blockIndex < histGridSize.x; blockIndex++) {
      for (int threadIndex = 0; threadIndex < histBlockSize.x; threadIndex++) {
        int index_data = blockIndex * histBlockSize.x + threadIndex;
        if (index_data < n) {
          int bin = (src[index_data] >> bit) & (nBins - 1);
          hist[blockIndex * nBins + bin]++;
        }
      }
    }
    /* timer.Stop(); */
    /* printf("Time: %.3f ms\n", timer.Elapsed()); */

    // Step 2: Scan (exclusive) "hist"
    /* printf(" + Step 2. Exclusive scan. "); */

    /* timer.Start(); */
    int pre = 0;
    for (int bin = 0; bin < nBins; bin++) {
      for (int blockIndex = 0; blockIndex < histGridSize.x; blockIndex++) {

        histScan[blockIndex * nBins + bin] = pre;
        pre = pre + hist[blockIndex * nBins + bin];
      }
    }

    /* timer.Stop(); */
    /* printf("Time: %.3f ms\n", timer.Elapsed()); */

    // Step 3: Scatter
    /* printf(" + Step 3. Scatter. "); */

    /* timer.Start(); */
    for (int blockIndex = 0; blockIndex < histGridSize.x; blockIndex++) {
      for (int threadIndex = 0; threadIndex < histBlockSize.x; threadIndex++) {
        int index_data = blockIndex * histBlockSize.x + threadIndex;
        if (index_data < n) {
          int bin =
              blockIndex * nBins + ((src[index_data] >> bit) & (nBins - 1));
          dst[histScan[bin]] = src[index_data];
          histScan[bin]++;
        }
      }
    }
    /* timer.Stop(); */
    /* printf("Time: %.3f ms\n", timer.Elapsed()); */

    // Swap "src" and "dst"
    uint32_t *tmp = src;
    src = dst;
    dst = tmp;
  }

  // Copy result to "out"
  memcpy(out, src, n * sizeof(uint32_t));

  // Free memories
  free(hist);
  free(histScan);
  free(originalSrc);
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
    printf("\nRadix sort by host (#paper: sequential radix sort)\n");
    sortByHost(in, n, out, nBits, blockSizes);
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

void checkCorrectness(uint32_t *out, uint32_t *correctOut, int n) {
  for (int i = 0; i < n; i++) {
    if (out[i] != correctOut[i]) {
      printf("INCORRECT :(\n");
      return;
    }
  }
  printf("CORRECT :)\n");
}

int main(int argc, char **argv) {
  // PRINT OUT DEVICE INFO
  printDeviceInfo();

  // SET UP INPUT SIZE
  int n = (1 << 24) + 1;
  // int n = 10;
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
