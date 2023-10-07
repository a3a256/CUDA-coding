#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <algorithm>

int relu_activate(int x){
    return std::max(0, x);
}

cudaError_t relu(int *a, int *b, int n);

__global__ void relu_kernel(int* a, int*b, int (*func)(int)){
    int i = threadIdx.x;
    b[i] = func(a[i]);
}

int main(){
    int a[5] = {1, -2, 3, 4, -7};
    int n = 5;
    int b[5] = {0};
    cudaError_t status = relu(a, b, n);
    for(int i=0; i<n; i++){
        printf("%d ", b[i]);
    }
    printf("\n");

    status = cudaDeviceReset();
    if(status != cudaSuccess){
        fprintf(stderr, "Failed to reset device\n");
        return 0;
    }
    return 1;
}

cudaError_t relu(int *a, int *b, int n){
    int *dev_a = 0;
    int *dev_b = 0;
    cudaError_t status;

    status = cudaSetDevice(0);
    if(status != cudaSuccess){
        fprintf(stderr, "Failed to set device, check whether you have CUDA device or not\n");
        goto Error;
    }

    status = cudaMalloc((void**)&dev_a, n*sizeof(int));
    if(status != cudaSuccess){
        fprintf(stderr, "Failed to allocate memory\n");
        goto Error;
    }

    status = cudaMalloc((void**)&dev_b, n*sizeof(int));
    if(status != cudaSuccess){
        fprintf(stderr, "Failed to allocate memory\n");
        goto Error;
    }

    status = cudaMemcpy(dev_a, a, n*sizeof(int), cudaMemcpyHostToDevice);
    if(status != cudaSuccess){
        fprintf(stderr, "Failed to copy from Host to GPU Buffer\n");
        goto Error;
    }

    relu_kernel<<<1,n>>>(dev_a, dev_b, &relu_activate);

    status = cudaGetLastError();
    if(status != cudaSuccess){
        fprintf(stderr, "Error occured: %s\n", cudaGetErrorString(status));
        goto Error;
    }

    status = cudaDeviceSynchronize();
    if(status != cudaSuccess){
        fprintf(stderr, "Failed to synchronize device, error no: %d\n", status);
        goto Error;
    }

    status = cudaMemcpy(b, dev_b, n*sizeof(int), cudaMemcpyDeviceToHost);

    if(status != cudaSuccess){
        fprintf(stderr, "Failed to copy from GPU Buffer to Host memory buffer\n");
        goto Error;
    }

    Error:
    cudaFree(dev_a);
    cudaFree(dev_b);

    return status;
}
