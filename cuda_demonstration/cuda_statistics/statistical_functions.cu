#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdio>

cudaError_t cudaMean1d(int* a, int *b, const int size);

__global__ void sum1dKernel(int *a, int *val) {
	int i = threadIdx.x;
	*val += a[i];
	printf("%d %d\n", a[i], *val);
}

int main() {
	const int m = 5;
	int arr[m] = { 1, 2, 3, 4, 5 };
	int res = -1;

	cudaError_t status = cudaMean1d(arr, &res, m);
	printf("\n");

	if (status != cudaSuccess) {
		fprintf(stderr, "Something went wrong\n");
		return 1;
	}

	printf("Mean value of a {1, 2, 3, 4, 5} array is %d\n", res);

	status = cudaDeviceReset();
	if (status != cudaSuccess) {
		fprintf(stderr, "Failed to reset device\n");
		return 1;
	}

	return 0;
}

cudaError_t cudaMean1d(int* a, int* b, const int size) {
	int *dev_a = 0;
	int *dev_c = 0;

	cudaError_t status;

	status = cudaSetDevice(0);
	if (status != cudaSuccess) {
		fprintf(stderr, "Failed to start device. Do you have CUDA device installed?\n");
		goto Error;
	}

	status = cudaMalloc((void**)&dev_a, size * sizeof(int));

	if (status != cudaSuccess) {
		fprintf(stderr, "Failed to allocate memory\n");
		goto Error;
	}

	status = cudaMalloc((void**)&dev_c, sizeof(int));

	if (status != cudaSuccess) {
		fprintf(stderr, "Failed to allocate memory\n");
		goto Error;
	}

	status = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
	if (status != cudaSuccess) {
		fprintf(stderr, "Failed to copy from host to GPU buffer\n");
		goto Error;
	}

	sum1dKernel << <1, size >> > (dev_a, dev_c);

	status = cudaGetLastError();
	if (status != cudaSuccess) {
		fprintf(stderr, "Last error message %s\n", cudaGetErrorString(status));
		goto Error;
	}

	status = cudaDeviceSynchronize();
	if (status != cudaSuccess) {
		fprintf(stderr, "Failed to synchronize device. Error %d\n", status);
		goto Error;
	}

	status = cudaMemcpy(b, dev_c, sizeof(int), cudaMemcpyDeviceToHost);
	if (status != cudaSuccess) {
		fprintf(stderr, "Failed to copy from GPU buffer to host\n");
		goto Error;
	}

Error:
	cudaFree(dev_a);
	return status;
}