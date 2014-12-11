#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "initializer.h"

#include <stdio.h>

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main()
{
	cudaError_t cudaStatus;
	unsigned int* dev_p = 0;
	unsigned int* dev_t = 0;

	cudaStatus = initialize(&dev_p, &dev_t);
	if (cudaStatus != cudaSuccess) {
		goto CLEANUP;
	}

	printf("DONE\n");

CLEANUP:
	cudaFree(dev_p);

	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	return 0;
}

/*

// Launch a kernel on the GPU with one thread for each element.
addKernel <<<1, size>>>(dev_c, dev_a, dev_b);
*/
