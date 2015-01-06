#ifndef INITIALIZER_H
#define INITIALIZER_H

#include <stdio.h>
#include "cuda_runtime.h"
#include "polynom.h"
#include "triangle.h"
#include "utils.h"

cudaError_t initialize(unsigned int** dev_p, uint3** dev_t, bool** dev_r) {
	// Set Device
	info("Device Initialization...\n");
	cudaError_t cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		return cudaStatus;
	}

	info("P compression...\n");
	unsigned int* p = compress_p();

	info("P device memory allocation...\n");
	cudaStatus = cudaMalloc((void**)dev_p, P_SIZE * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return cudaStatus;
	}

	info("P copying to device...\n");
	cudaStatus = cudaMemcpy(*dev_p, p, P_SIZE * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		return cudaStatus;
	}

	info("T device memory allocation...\n");
	cudaStatus = cudaMalloc((void**)dev_t, T_SIZE * T_V_SIZE * sizeof(uint3));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return cudaStatus;
	}

	info("T compying to device...\n");
	cudaStatus = cudaMemcpy(*dev_t, t, T_SIZE * T_V_SIZE * sizeof(uint3), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		return cudaStatus;
	}

	info("R create empty element on the device...\n");
	cudaStatus = cudaMalloc((void**)dev_r, sizeof(bool));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		return cudaStatus;
	}

	info("R initialize...\n");
	bool b = false;
	cudaStatus = cudaMemcpy(*dev_r, &b, sizeof(bool), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		return cudaStatus;
	}

	return cudaSuccess;
}

#endif