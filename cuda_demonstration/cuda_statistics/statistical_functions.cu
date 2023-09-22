#include <stdio.h>
#include <math.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdio>

#define SIZE 12
#define SHMEMESIZE 12*4

cudaError_t cudaMean1d(int* a, int *b, const int m, const int n, dim3 threads, dim3 blocks);

// yet to test sum1d kerel for sum reduction

__global__ void sum1d(int* a, int* b, const int size) {
	__shared__ int partial_sum[SHMEMESIZE];

	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	partial_sum[threadIdx.x] = a[tid];
	__synchthreads();

	for (int s = 1; s < blockDim.x; s *= 2) {
		if (threadIdx.x % (2 * s) == 0) {
			partial_sum[threadIdx.x] += partial_sum[threadIdx.x + s];
		}
		__syncthreads();
	}

	if (threadIdx.x == 0) {
		b[blockIdx.x] = partial_sum[0];
	}
}

__global__ void sum1dKernel(int *a, int *val, const int m, const int n) {
	int row = blockDim.y * blockIdx.y + threadIdx.y;
	int col = blockDim.x * blockIdx.x + threadIdx.x;
	if (row < m && col < n) {
		printf("%d ", n * row + col);
		val[col] = a[n * row + col];
		printf("part %d %d \n", col, val[col]);
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

	int threads = 1;
	int blocks = (m * n + threads - 1) / threads;

	dim3 THREADS(threads, threads);
	dim3 BLOCKS(blocks, blocks);

	cudaError_t status = cudaMean1d(a, res, m, n, THREADS, BLOCKS);
	printf("\n");

	if (status != cudaSuccess) {
		fprintf(stderr, "Something went wrong\n");
		return 1;
	}

	
	for (int i = 0; i < n; i++) {
		printf("%d ", res[i]);
	}
	printf("\n");

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
	
	status = cudaMemcpy(dev_c, b, n * sizeof(int), cudaMemcpyHostToDevice);
	if (status != cudaSuccess) {
		fprintf(stderr, "Failed to copy from host to GPU buffer\n");
		goto Error;
	}

	sum1dKernel << <blocks, threads >> > (dev_a, dev_c, m, n);

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
	cudaFree(dev_c);
	return status;
}