#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdio>

cudaError_t cudaMean1d(int* a, int *b, const int m, const int n, dim3 threads, dim3 blocks);

__global__ void sum1dKernel(int *a, int *val, const int m, const int n) {
	int row = blockDim.y * blockIdx.y + threadIdx.y;
	int col = blockDim.x * blockIdx.x + threadIdx.x;
	if (col < n && row < m) {
		val[col] = a[n * row + col];
	}
}

int main() {
	const int m = 4;
	const int n = 3;
	int arr[m][n] = { {1, 2, 3}, {4, 2, 6}, {7, 2, 9}, {9, 5, 5} };
	int res[n] = { 0 };

	int a[m * n];
	int i, j, k;
	k = 0;
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			a[k] = arr[i][j];
			k++;
		}
	}

	cudaError_t status = cudaMean1d(a, res, m, n);
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

cudaError_t cudaMean1d(int* a, int* b, const int m, const int n, dim3 threads, dim3 blocks) {
	int *dev_a = 0;
	int *dev_c = 0;

	cudaError_t status;

	status = cudaSetDevice(0);
	if (status != cudaSuccess) {
		fprintf(stderr, "Failed to start device. Do you have CUDA device installed?\n");
		goto Error;
	}

	status = cudaMalloc((void**)&dev_a, m * n * sizeof(int));

	if (status != cudaSuccess) {
		fprintf(stderr, "Failed to allocate memory\n");
		goto Error;
	}

	status = cudaMalloc((void**)&dev_c, n * sizeof(int));

	if (status != cudaSuccess) {
		fprintf(stderr, "Failed to allocate memory\n");
		goto Error;
	}

	status = cudaMemcpy(dev_a, a, m * n * sizeof(int), cudaMemcpyHostToDevice);
	if (status != cudaSuccess) {
		fprintf(stderr, "Failed to copy from host to GPU buffer\n");
		goto Error;
	}

	sum1dKernel << <blocks, threads >> > (dev_a, dev_c);

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