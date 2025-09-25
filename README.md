# PCA: EXP-1  SUM ARRAY GPU
<h3>ENTER YOUR NAME: VINCY JOVITHA V</h3>
<h3>ENTER YOUR REGISTER NO: 212223230242</h3>
<h3>EX. NO 01</h3>
<h3>DATE 24/09/2025</h3>
<h1> <align=center> SUM ARRAY ON HOST AND DEVICE </h3>
PCA-GPU-based-vector-summation.-Explore-the-differences.
i) Using the program sumArraysOnGPU-timer.cu, set the block.x = 1023. Recompile and run it. Compare the result with the execution configuration of block.x = 1024. Try to explain the difference and the reason.

ii) Refer to sumArraysOnGPU-timer.cu, and let block.x = 256. Make a new kernel to let each thread handle two elements. Compare the results with other execution confi gurations.
## AIM:

To perform vector addition on host and device.

## EQUIPMENTS REQUIRED:
Hardware – PCs with NVIDIA GPU & CUDA NVCC
Google Colab with NVCC Compiler




## PROCEDURE:

1. Initialize the device and set the device properties.
2. Allocate memory on the host for input and output arrays.
3. Initialize input arrays with random values on the host.
4. Allocate memory on the device for input and output arrays, and copy input data from host to device.
5. Launch a CUDA kernel to perform vector addition on the device.
6. Copy output data from the device to the host and verify the results against the host's sequential vector addition. Free memory on the host and the device.

## PROGRAM:
```cpp
!pip install git+https://github.com/andreinechaev/nvcc4jupyter.git

!nvcc --version
!nvidia-smi

%%writefile vector_add.cu
#include <cuda_runtime.h>
#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <time.h>

// ================= Macros ==================
#define CHECK(call) { \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__); \
        fprintf(stderr, "code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1); \
    } \
}

inline double seconds() {
    struct timeval tp;
    struct timezone tzp;
    gettimeofday(&tp, &tzp);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

// ================= Utility Functions ==================
void checkResult(float *hostRef, float *gpuRef, const int N) {
    double epsilon = 1.0E-8;
    for (int i = 0; i < N; i++) {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon) {
            printf("Mismatch at %d: host %f gpu %f\n", i, hostRef[i], gpuRef[i]);
            return;
        }
    }
    printf("Arrays match.\n\n");
}

void initialData(float *ip, int size) {
    time_t t;
    srand((unsigned) time(&t));
    for (int i = 0; i < size; i++) {
        ip[i] = (float)( rand() & 0xFF ) / 10.0f;
    }
}

void sumArraysOnHost(float *A, float *B, float *C, const int N) {
    for (int idx = 0; idx < N; idx++) {
        C[idx] = A[idx] + B[idx];
    }
}

// ================= Kernel ==================
__global__ void sumArraysOnGPU(float *A, float *B, float *C, const int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
}

// ================= Main ==================
int main(int argc, char **argv) {
    printf("%s Starting...\n", argv[0]);

    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    int nElem = 1 << 20;  // smaller size for Colab
    size_t nBytes = nElem * sizeof(float);
    printf("Vector size %d\n", nElem);

    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A     = (float *)malloc(nBytes);
    h_B     = (float *)malloc(nBytes);
    hostRef = (float *)malloc(nBytes);
    gpuRef  = (float *)malloc(nBytes);

    double iStart, iElaps;
    iStart = seconds();
    initialData(h_A, nElem);
    initialData(h_B, nElem);
    iElaps = seconds() - iStart;
    printf("initialData Time elapsed %f sec\n", iElaps);

    memset(hostRef, 0, nBytes);
    memset(gpuRef,  0, nBytes);

    iStart = seconds();
    sumArraysOnHost(h_A, h_B, hostRef, nElem);
    iElaps = seconds() - iStart;
    printf("sumArraysOnHost Time elapsed %f sec\n", iElaps);

    float *d_A, *d_B, *d_C;
    CHECK(cudaMalloc((float**)&d_A, nBytes));
    CHECK(cudaMalloc((float**)&d_B, nBytes));
    CHECK(cudaMalloc((float**)&d_C, nBytes));

    CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_C, gpuRef, nBytes, cudaMemcpyHostToDevice));

    int iLen = 256;
    dim3 block (iLen);
    dim3 grid  ((nElem + block.x - 1) / block.x);

    iStart = seconds();
    sumArraysOnGPU<<<grid, block>>>(d_A, d_B, d_C, nElem);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    printf("sumArraysOnGPU <<<  %d, %d  >>>  Time elapsed %f sec\n", grid.x, block.x, iElaps);

    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
    checkResult(hostRef, gpuRef, nElem);

    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));
    free(h_A); free(h_B); free(hostRef); free(gpuRef);

    return(0);

}

// Compile and Run
!nvcc vector_add.cu -o vector_add
!./vector_add


```


## OUTPUT:
<img width="664" height="178" alt="image" src="https://github.com/user-attachments/assets/e22046ab-48c6-4c1a-bc81-a611d1e352fb" />


## RESULT:
Thus, Implementation of sum arrays on host and device is done in nvcc cuda using random number.
