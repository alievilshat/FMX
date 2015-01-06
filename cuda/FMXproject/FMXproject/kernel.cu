#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "initializer.h"
#include "utils.h"

#include <stdio.h>

#define GRID_SIZE 10000
#define BLOCK_SIZE 256
#define NUMBER_OF_CANDIDATES 1

#define msk(p, p_i, j) (j * (1 - ((p >> p_i) & 2)) * ((p >> p_i) & 1))

__host__ __device__ void calculate(short* c, unsigned int* d_p, bool* res) {
	short j0, j1, j2, j3, j4, j5, j6;

	//for (int i = 0; i++ < NUMBER_OF_CANDIDATES; nextcandidate(c)) 
	{

		unsigned int p1 = d_p[c[0]];
		unsigned int p2 = d_p[c[1]];
		unsigned int p3 = d_p[c[2]];
		unsigned int p4 = d_p[c[3]];
		unsigned int p5 = d_p[c[4]];
		unsigned int p6 = d_p[c[5]];
		unsigned int p7 = d_p[c[6]];

		//validate c1
		for (j6 = -1; j6 <= 1; j6++)
		for (j5 = -1; j5 <= 1; j5++)
		for (j4 = -1; j4 <= 1; j4++)
		for (j3 = -1; j3 <= 1; j3++)
		for (j2 = -1; j2 <= 1; j2++)
		for (j1 = -1; j1 <= 1; j1++)
		for (j0 = -1; j0 <= 1; j0++) {
			if ((1 /*ae*/ == msk(p1, 0, j6) + msk(p2, 0, j5) + msk(p3, 0, j4) + msk(p4, 0, j3) + msk(p5, 0, j2) + msk(p6, 0, j1) + msk(p7, 0, j0))
				&& (0 /*af*/ == msk(p1, 2, j6) + msk(p2, 2, j5) + msk(p3, 2, j4) + msk(p4, 2, j3) + msk(p5, 2, j2) + msk(p6, 2, j1) + msk(p7, 2, j0))
				&& (0 /*ag*/ == msk(p1, 4, j6) + msk(p2, 4, j5) + msk(p3, 4, j4) + msk(p4, 4, j3) + msk(p5, 4, j2) + msk(p6, 4, j1) + msk(p7, 4, j0))
				&& (0 /*ah*/ == msk(p1, 6, j6) + msk(p2, 6, j5) + msk(p3, 6, j4) + msk(p4, 6, j3) + msk(p5, 6, j2) + msk(p6, 6, j1) + msk(p7, 6, j0))
				&& (0 /*be*/ == msk(p1, 8, j6) + msk(p2, 8, j5) + msk(p3, 8, j4) + msk(p4, 8, j3) + msk(p5, 8, j2) + msk(p6, 8, j1) + msk(p7, 8, j0))
				&& (0 /*bf*/ == msk(p1, 10, j6) + msk(p2, 10, j5) + msk(p3, 10, j4) + msk(p4, 10, j3) + msk(p5, 10, j2) + msk(p6, 10, j1) + msk(p7, 10, j0))
				&& (1 /*bg*/ == msk(p1, 12, j6) + msk(p2, 12, j5) + msk(p3, 12, j4) + msk(p4, 12, j3) + msk(p5, 12, j2) + msk(p6, 12, j1) + msk(p7, 12, j0))
				&& (0 /*bh*/ == msk(p1, 14, j6) + msk(p2, 14, j5) + msk(p3, 14, j4) + msk(p4, 14, j3) + msk(p5, 14, j2) + msk(p6, 14, j1) + msk(p7, 14, j0))
				&& (0 /*ce*/ == msk(p1, 16, j6) + msk(p2, 16, j5) + msk(p3, 16, j4) + msk(p4, 16, j3) + msk(p5, 16, j2) + msk(p6, 16, j1) + msk(p7, 16, j0))
				&& (0 /*cf*/ == msk(p1, 18, j6) + msk(p2, 18, j5) + msk(p3, 18, j4) + msk(p4, 18, j3) + msk(p5, 18, j2) + msk(p6, 18, j1) + msk(p7, 18, j0))
				&& (0 /*cg*/ == msk(p1, 20, j6) + msk(p2, 20, j5) + msk(p3, 20, j4) + msk(p4, 20, j3) + msk(p5, 20, j2) + msk(p6, 20, j1) + msk(p7, 20, j0))
				&& (0 /*ch*/ == msk(p1, 22, j6) + msk(p2, 22, j5) + msk(p3, 22, j4) + msk(p4, 22, j3) + msk(p5, 22, j2) + msk(p6, 22, j1) + msk(p7, 22, j0))
				&& (0 /*de*/ == msk(p1, 24, j6) + msk(p2, 24, j5) + msk(p3, 24, j4) + msk(p4, 24, j3) + msk(p5, 24, j2) + msk(p6, 24, j1) + msk(p7, 24, j0))
				&& (0 /*df*/ == msk(p1, 26, j6) + msk(p2, 26, j5) + msk(p3, 26, j4) + msk(p4, 26, j3) + msk(p5, 26, j2) + msk(p6, 26, j1) + msk(p7, 26, j0))
				&& (0 /*dg*/ == msk(p1, 28, j6) + msk(p2, 28, j5) + msk(p3, 28, j4) + msk(p4, 28, j3) + msk(p5, 28, j2) + msk(p6, 28, j1) + msk(p7, 28, j0))
				&& (0 /*dh*/ == msk(p1, 30, j6) + msk(p2, 30, j5) + msk(p3, 30, j4) + msk(p4, 30, j3) + msk(p5, 30, j2) + msk(p6, 30, j1) + msk(p7, 30, j0)))
			{
				goto c2;
			}
		}
		return;// continue;

	c2: // validate c2
		for (j6 = -1; j6 <= 1; j6++)
		for (j5 = -1; j5 <= 1; j5++)
		for (j4 = -1; j4 <= 1; j4++)
		for (j3 = -1; j3 <= 1; j3++)
		for (j2 = -1; j2 <= 1; j2++)
		for (j1 = -1; j1 <= 1; j1++)
		for (j0 = -1; j0 <= 1; j0++) {
			if ((0 /*ae*/ == msk(p1, 0, j6) + msk(p2, 0, j5) + msk(p3, 0, j4) + msk(p4, 0, j3) + msk(p5, 0, j2) + msk(p6, 0, j1) + msk(p7, 0, j0))
				&& (1 /*af*/ == msk(p1, 2, j6) + msk(p2, 2, j5) + msk(p3, 2, j4) + msk(p4, 2, j3) + msk(p5, 2, j2) + msk(p6, 2, j1) + msk(p7, 2, j0))
				&& (0 /*ag*/ == msk(p1, 4, j6) + msk(p2, 4, j5) + msk(p3, 4, j4) + msk(p4, 4, j3) + msk(p5, 4, j2) + msk(p6, 4, j1) + msk(p7, 4, j0))
				&& (0 /*ah*/ == msk(p1, 6, j6) + msk(p2, 6, j5) + msk(p3, 6, j4) + msk(p4, 6, j3) + msk(p5, 6, j2) + msk(p6, 6, j1) + msk(p7, 6, j0))
				&& (0 /*be*/ == msk(p1, 8, j6) + msk(p2, 8, j5) + msk(p3, 8, j4) + msk(p4, 8, j3) + msk(p5, 8, j2) + msk(p6, 8, j1) + msk(p7, 8, j0))
				&& (0 /*bf*/ == msk(p1, 10, j6) + msk(p2, 10, j5) + msk(p3, 10, j4) + msk(p4, 10, j3) + msk(p5, 10, j2) + msk(p6, 10, j1) + msk(p7, 10, j0))
				&& (0 /*bg*/ == msk(p1, 12, j6) + msk(p2, 12, j5) + msk(p3, 12, j4) + msk(p4, 12, j3) + msk(p5, 12, j2) + msk(p6, 12, j1) + msk(p7, 12, j0))
				&& (1 /*bh*/ == msk(p1, 14, j6) + msk(p2, 14, j5) + msk(p3, 14, j4) + msk(p4, 14, j3) + msk(p5, 14, j2) + msk(p6, 14, j1) + msk(p7, 14, j0))
				&& (0 /*ce*/ == msk(p1, 16, j6) + msk(p2, 16, j5) + msk(p3, 16, j4) + msk(p4, 16, j3) + msk(p5, 16, j2) + msk(p6, 16, j1) + msk(p7, 16, j0))
				&& (0 /*cf*/ == msk(p1, 18, j6) + msk(p2, 18, j5) + msk(p3, 18, j4) + msk(p4, 18, j3) + msk(p5, 18, j2) + msk(p6, 18, j1) + msk(p7, 18, j0))
				&& (0 /*cg*/ == msk(p1, 20, j6) + msk(p2, 20, j5) + msk(p3, 20, j4) + msk(p4, 20, j3) + msk(p5, 20, j2) + msk(p6, 20, j1) + msk(p7, 20, j0))
				&& (0 /*ch*/ == msk(p1, 22, j6) + msk(p2, 22, j5) + msk(p3, 22, j4) + msk(p4, 22, j3) + msk(p5, 22, j2) + msk(p6, 22, j1) + msk(p7, 22, j0))
				&& (0 /*de*/ == msk(p1, 24, j6) + msk(p2, 24, j5) + msk(p3, 24, j4) + msk(p4, 24, j3) + msk(p5, 24, j2) + msk(p6, 24, j1) + msk(p7, 24, j0))
				&& (0 /*df*/ == msk(p1, 26, j6) + msk(p2, 26, j5) + msk(p3, 26, j4) + msk(p4, 26, j3) + msk(p5, 26, j2) + msk(p6, 26, j1) + msk(p7, 26, j0))
				&& (0 /*dg*/ == msk(p1, 28, j6) + msk(p2, 28, j5) + msk(p3, 28, j4) + msk(p4, 28, j3) + msk(p5, 28, j2) + msk(p6, 28, j1) + msk(p7, 28, j0))
				&& (0 /*dh*/ == msk(p1, 30, j6) + msk(p2, 30, j5) + msk(p3, 30, j4) + msk(p4, 30, j3) + msk(p5, 30, j2) + msk(p6, 30, j1) + msk(p7, 30, j0)))
			{
				goto c3;
			}
		}
		return;// continue;

	c3: // validate c3
		for (j6 = -1; j6 <= 1; j6++)
		for (j5 = -1; j5 <= 1; j5++)
		for (j4 = -1; j4 <= 1; j4++)
		for (j3 = -1; j3 <= 1; j3++)
		for (j2 = -1; j2 <= 1; j2++)
		for (j1 = -1; j1 <= 1; j1++)
		for (j0 = -1; j0 <= 1; j0++) {
			if ((0 /*ae*/ == msk(p1, 0, j6) + msk(p2, 0, j5) + msk(p3, 0, j4) + msk(p4, 0, j3) + msk(p5, 0, j2) + msk(p6, 0, j1) + msk(p7, 0, j0))
				&& (0 /*af*/ == msk(p1, 2, j6) + msk(p2, 2, j5) + msk(p3, 2, j4) + msk(p4, 2, j3) + msk(p5, 2, j2) + msk(p6, 2, j1) + msk(p7, 2, j0))
				&& (0 /*ag*/ == msk(p1, 4, j6) + msk(p2, 4, j5) + msk(p3, 4, j4) + msk(p4, 4, j3) + msk(p5, 4, j2) + msk(p6, 4, j1) + msk(p7, 4, j0))
				&& (0 /*ah*/ == msk(p1, 6, j6) + msk(p2, 6, j5) + msk(p3, 6, j4) + msk(p4, 6, j3) + msk(p5, 6, j2) + msk(p6, 6, j1) + msk(p7, 6, j0))
				&& (0 /*be*/ == msk(p1, 8, j6) + msk(p2, 8, j5) + msk(p3, 8, j4) + msk(p4, 8, j3) + msk(p5, 8, j2) + msk(p6, 8, j1) + msk(p7, 8, j0))
				&& (0 /*bf*/ == msk(p1, 10, j6) + msk(p2, 10, j5) + msk(p3, 10, j4) + msk(p4, 10, j3) + msk(p5, 10, j2) + msk(p6, 10, j1) + msk(p7, 10, j0))
				&& (0 /*bg*/ == msk(p1, 12, j6) + msk(p2, 12, j5) + msk(p3, 12, j4) + msk(p4, 12, j3) + msk(p5, 12, j2) + msk(p6, 12, j1) + msk(p7, 12, j0))
				&& (0 /*bh*/ == msk(p1, 14, j6) + msk(p2, 14, j5) + msk(p3, 14, j4) + msk(p4, 14, j3) + msk(p5, 14, j2) + msk(p6, 14, j1) + msk(p7, 14, j0))
				&& (1 /*ce*/ == msk(p1, 16, j6) + msk(p2, 16, j5) + msk(p3, 16, j4) + msk(p4, 16, j3) + msk(p5, 16, j2) + msk(p6, 16, j1) + msk(p7, 16, j0))
				&& (0 /*cf*/ == msk(p1, 18, j6) + msk(p2, 18, j5) + msk(p3, 18, j4) + msk(p4, 18, j3) + msk(p5, 18, j2) + msk(p6, 18, j1) + msk(p7, 18, j0))
				&& (0 /*cg*/ == msk(p1, 20, j6) + msk(p2, 20, j5) + msk(p3, 20, j4) + msk(p4, 20, j3) + msk(p5, 20, j2) + msk(p6, 20, j1) + msk(p7, 20, j0))
				&& (0 /*ch*/ == msk(p1, 22, j6) + msk(p2, 22, j5) + msk(p3, 22, j4) + msk(p4, 22, j3) + msk(p5, 22, j2) + msk(p6, 22, j1) + msk(p7, 22, j0))
				&& (0 /*de*/ == msk(p1, 24, j6) + msk(p2, 24, j5) + msk(p3, 24, j4) + msk(p4, 24, j3) + msk(p5, 24, j2) + msk(p6, 24, j1) + msk(p7, 24, j0))
				&& (0 /*df*/ == msk(p1, 26, j6) + msk(p2, 26, j5) + msk(p3, 26, j4) + msk(p4, 26, j3) + msk(p5, 26, j2) + msk(p6, 26, j1) + msk(p7, 26, j0))
				&& (1 /*dg*/ == msk(p1, 28, j6) + msk(p2, 28, j5) + msk(p3, 28, j4) + msk(p4, 28, j3) + msk(p5, 28, j2) + msk(p6, 28, j1) + msk(p7, 28, j0))
				&& (0 /*dh*/ == msk(p1, 30, j6) + msk(p2, 30, j5) + msk(p3, 30, j4) + msk(p4, 30, j3) + msk(p5, 30, j2) + msk(p6, 30, j1) + msk(p7, 30, j0)))
			{
				goto c4;
			}
		}
		return;// continue;

	c4: // validate c4
		for (j6 = -1; j6 <= 1; j6++)
		for (j5 = -1; j5 <= 1; j5++)
		for (j4 = -1; j4 <= 1; j4++)
		for (j3 = -1; j3 <= 1; j3++)
		for (j2 = -1; j2 <= 1; j2++)
		for (j1 = -1; j1 <= 1; j1++)
		for (j0 = -1; j0 <= 1; j0++) {
			if ((0 /*ae*/ == msk(p1, 0, j6) + msk(p2, 0, j5) + msk(p3, 0, j4) + msk(p4, 0, j3) + msk(p5, 0, j2) + msk(p6, 0, j1) + msk(p7, 0, j0))
				&& (0 /*af*/ == msk(p1, 2, j6) + msk(p2, 2, j5) + msk(p3, 2, j4) + msk(p4, 2, j3) + msk(p5, 2, j2) + msk(p6, 2, j1) + msk(p7, 2, j0))
				&& (0 /*ag*/ == msk(p1, 4, j6) + msk(p2, 4, j5) + msk(p3, 4, j4) + msk(p4, 4, j3) + msk(p5, 4, j2) + msk(p6, 4, j1) + msk(p7, 4, j0))
				&& (0 /*ah*/ == msk(p1, 6, j6) + msk(p2, 6, j5) + msk(p3, 6, j4) + msk(p4, 6, j3) + msk(p5, 6, j2) + msk(p6, 6, j1) + msk(p7, 6, j0))
				&& (0 /*be*/ == msk(p1, 8, j6) + msk(p2, 8, j5) + msk(p3, 8, j4) + msk(p4, 8, j3) + msk(p5, 8, j2) + msk(p6, 8, j1) + msk(p7, 8, j0))
				&& (0 /*bf*/ == msk(p1, 10, j6) + msk(p2, 10, j5) + msk(p3, 10, j4) + msk(p4, 10, j3) + msk(p5, 10, j2) + msk(p6, 10, j1) + msk(p7, 10, j0))
				&& (0 /*bg*/ == msk(p1, 12, j6) + msk(p2, 12, j5) + msk(p3, 12, j4) + msk(p4, 12, j3) + msk(p5, 12, j2) + msk(p6, 12, j1) + msk(p7, 12, j0))
				&& (0 /*bh*/ == msk(p1, 14, j6) + msk(p2, 14, j5) + msk(p3, 14, j4) + msk(p4, 14, j3) + msk(p5, 14, j2) + msk(p6, 14, j1) + msk(p7, 14, j0))
				&& (0 /*ce*/ == msk(p1, 16, j6) + msk(p2, 16, j5) + msk(p3, 16, j4) + msk(p4, 16, j3) + msk(p5, 16, j2) + msk(p6, 16, j1) + msk(p7, 16, j0))
				&& (1 /*cf*/ == msk(p1, 18, j6) + msk(p2, 18, j5) + msk(p3, 18, j4) + msk(p4, 18, j3) + msk(p5, 18, j2) + msk(p6, 18, j1) + msk(p7, 18, j0))
				&& (0 /*cg*/ == msk(p1, 20, j6) + msk(p2, 20, j5) + msk(p3, 20, j4) + msk(p4, 20, j3) + msk(p5, 20, j2) + msk(p6, 20, j1) + msk(p7, 20, j0))
				&& (0 /*ch*/ == msk(p1, 22, j6) + msk(p2, 22, j5) + msk(p3, 22, j4) + msk(p4, 22, j3) + msk(p5, 22, j2) + msk(p6, 22, j1) + msk(p7, 22, j0))
				&& (0 /*de*/ == msk(p1, 24, j6) + msk(p2, 24, j5) + msk(p3, 24, j4) + msk(p4, 24, j3) + msk(p5, 24, j2) + msk(p6, 24, j1) + msk(p7, 24, j0))
				&& (0 /*df*/ == msk(p1, 26, j6) + msk(p2, 26, j5) + msk(p3, 26, j4) + msk(p4, 26, j3) + msk(p5, 26, j2) + msk(p6, 26, j1) + msk(p7, 26, j0))
				&& (0 /*dg*/ == msk(p1, 28, j6) + msk(p2, 28, j5) + msk(p3, 28, j4) + msk(p4, 28, j3) + msk(p5, 28, j2) + msk(p6, 28, j1) + msk(p7, 28, j0))
				&& (1 /*dh*/ == msk(p1, 30, j6) + msk(p2, 30, j5) + msk(p3, 30, j4) + msk(p4, 30, j3) + msk(p5, 30, j2) + msk(p6, 30, j1) + msk(p7, 30, j0)))
			{
				*res = true;
				return;
			}
		}
	}
}

__global__ void g_calculate(unsigned int* d_p, uint3* d_t, uint3 start, bool* res) {

	short c[7];
	unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;

	getcandidate(d_t, uint3_add(start, make_uint3(id, 0, 0)), c);
	calculate(c, d_p, res);
}

__global__ void g_getcandidateindex(uint3* d_t, short* res, uint3* n)
{
	getcandidateindex(d_t, res, n);
}

__global__ void g_getcandidate(uint3* d_t, uint3 n, short* res)
{
	getcandidate(d_t, n, res);
}

int main()
{
	cudaError_t cudaStatus;
	unsigned int* dev_p = 0;
	uint3* dev_t = 0;
	bool* dev_r = 0;
	bool r = false;

	cudaStatus = initialize(&dev_p, &dev_t, &dev_r);
	if (cudaStatus != cudaSuccess) {
		goto CLEANUP;
	}

	info("START:\n");

	uint3 max = make_uint3(0x1a451e22, 0x4823143b, 0x25);
	uint3 inc = make_uint3(GRID_SIZE * BLOCK_SIZE * NUMBER_OF_CANDIDATES, 0, 0);
	uint3 start = make_uint3(0, 0, 0);

	uint3 strassen = make_uint3(3261699961, 1784383582, 4); // { 57, 160, 350, 1050, 1311, 1771, 2961 }

	for (uint3 n = start; uint3_cmp(n, max) < 0; n = add(n, inc)) {

		g_calculate << <GRID_SIZE, BLOCK_SIZE>> >(dev_p, dev_t, n, dev_r);

		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			info("(%u, %u, %u): cuda execution failed!\n", n.x, n.y, n.z);
			goto CLEANUP;
		}

		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			info("(%u, %u, %u): cudaDeviceSynchronize returned error code %d after launching addKernel!\n", n.x, n.y, n.z, cudaStatus);
			goto CLEANUP;
		}

		cudaStatus = cudaMemcpy(&r, dev_r, sizeof(bool), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			info("(%u, %u, %u): cudaMemcpy failed!\n", n.x, n.y, n.z);
			goto CLEANUP;
		}

		info("(%u, %u, %u): %s\n", n.x, n.y, n.z, r ? "TRUE" : "FALSE");
		if (r) break;
	}

CLEANUP:
	cudaFree(dev_p);

	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	return 0;
}