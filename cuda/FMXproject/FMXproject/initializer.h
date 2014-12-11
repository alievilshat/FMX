#ifndef INITIALIZER_H
#define INITIALIZER_H

#include "cuda_runtime.h"
#include "polynom.h"
#include "triangle.h"
#include "utils.h"
#include <stdio.h>

cudaError_t initialize(unsigned int **dev_p, unsigned int **dev_t) {
	// Set Device
	info("Device Initialization: ");
	cudaError_t cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		return cudaStatus;
	}
	info("OK\n");

	info("P compression: ");
	unsigned int* p = compress_p();
	info("OK\n");

	info("P device memory allocation: ");
	cudaStatus = cudaMalloc((void**)dev_p, P_SIZE * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return cudaStatus;
	}
	info("OK\n");

	info("P copying to device: ");
	cudaStatus = cudaMemcpy(*dev_p, p, P_SIZE * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		return cudaStatus;
	}
	info("OK\n");

	info("T device memory allocation: ");
	cudaStatus = cudaMalloc((void**)dev_t, T_SIZE * P_SIZE * sizeof(uint3));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return cudaStatus;
	}
	info("OK\n");

	info("T compying to device: ");
	cudaStatus = cudaMemcpy(*dev_t, t, T_SIZE * P_SIZE * sizeof(uint3), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		return cudaStatus;
	}
	info("OK\n");

	return cudaSuccess;
}

#endif