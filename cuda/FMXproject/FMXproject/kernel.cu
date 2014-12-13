#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "initializer.h"

#include <stdio.h>

#define bit(e, i) ((e >> i) & 1)
#define bit2(e, i) ((e >> (i-1)) & 2)

__global__ void calculate(unsigned int* d_p, uint3** d_t, uint3 start, bool* res) {
	char j;
	short* c = getcandidate(d_t, start);

	unsigned int p1 = d_p[c[0]];
	unsigned int p2 = d_p[c[1]];
	unsigned int p3 = d_p[c[2]];
	unsigned int p4 = d_p[c[3]];
	unsigned int p5 = d_p[c[4]];
	unsigned int p6 = d_p[c[5]];
	unsigned int p7 = d_p[c[6]];

	*res = false;
	
	//validate c1
	for (j = 0; j < 128; j++)
	if ((1 /*ae*/ == (1 - bit2(p1, 1)) * bit(p1, 0) * bit(j, 6) + (1 - bit2(p2, 1)) * bit(p2, 0) * bit(j, 5) + (1 - bit2(p3, 1)) * bit(p3, 0) * bit(j, 4) + (1 - bit2(p4, 1)) * bit(p4, 0) * bit(j, 3) + (1 - bit2(p5, 1)) * bit(p5, 0) * bit(j, 2) + (1 - bit2(p6, 1)) * bit(p6, 0) * bit(j, 1) + (1 - bit2(p7, 1)) * bit(p7, 0) * bit(j, 0))
		&& (0 /*af*/ == (1 - bit2(p1, 3)) * bit(p1, 2) * bit(j, 6) + (1 - bit2(p2, 3)) * bit(p2, 2) * bit(j, 5) + (1 - bit2(p3, 3)) * bit(p3, 2) * bit(j, 4) + (1 - bit2(p4, 3)) * bit(p4, 2) * bit(j, 3) + (1 - bit2(p5, 3)) * bit(p5, 2) * bit(j, 2) + (1 - bit2(p6, 3)) * bit(p6, 2) * bit(j, 1) + (1 - bit2(p7, 3)) * bit(p7, 2) * bit(j, 0))
		&& (0 /*ag*/ == (1 - bit2(p1, 5)) * bit(p1, 4) * bit(j, 6) + (1 - bit2(p2, 5)) * bit(p2, 4) * bit(j, 5) + (1 - bit2(p3, 5)) * bit(p3, 4) * bit(j, 4) + (1 - bit2(p4, 5)) * bit(p4, 4) * bit(j, 3) + (1 - bit2(p5, 5)) * bit(p5, 4) * bit(j, 2) + (1 - bit2(p6, 5)) * bit(p6, 4) * bit(j, 1) + (1 - bit2(p7, 5)) * bit(p7, 4) * bit(j, 0))
		&& (0 /*ah*/ == (1 - bit2(p1, 7)) * bit(p1, 6) * bit(j, 6) + (1 - bit2(p2, 7)) * bit(p2, 6) * bit(j, 5) + (1 - bit2(p3, 7)) * bit(p3, 6) * bit(j, 4) + (1 - bit2(p4, 7)) * bit(p4, 6) * bit(j, 3) + (1 - bit2(p5, 7)) * bit(p5, 6) * bit(j, 2) + (1 - bit2(p6, 7)) * bit(p6, 6) * bit(j, 1) + (1 - bit2(p7, 7)) * bit(p7, 6) * bit(j, 0))
		&& (0 /*be*/ == (1 - bit2(p1, 9)) * bit(p1, 8) * bit(j, 6) + (1 - bit2(p2, 9)) * bit(p2, 8) * bit(j, 5) + (1 - bit2(p3, 9)) * bit(p3, 8) * bit(j, 4) + (1 - bit2(p4, 9)) * bit(p4, 8) * bit(j, 3) + (1 - bit2(p5, 9)) * bit(p5, 8) * bit(j, 2) + (1 - bit2(p6, 9)) * bit(p6, 8) * bit(j, 1) + (1 - bit2(p7, 9)) * bit(p7, 8) * bit(j, 0))
		&& (0 /*bf*/ == (1 - bit2(p1, 11)) * bit(p1, 10) * bit(j, 6) + (1 - bit2(p2, 11)) * bit(p2, 10) * bit(j, 5) + (1 - bit2(p3, 11)) * bit(p3, 10) * bit(j, 4) + (1 - bit2(p4, 11)) * bit(p4, 10) * bit(j, 3) + (1 - bit2(p5, 11)) * bit(p5, 10) * bit(j, 2) + (1 - bit2(p6, 11)) * bit(p6, 10) * bit(j, 1) + (1 - bit2(p7, 11)) * bit(p7, 10) * bit(j, 0))
		&& (1 /*bg*/ == (1 - bit2(p1, 13)) * bit(p1, 12) * bit(j, 6) + (1 - bit2(p2, 13)) * bit(p2, 12) * bit(j, 5) + (1 - bit2(p3, 13)) * bit(p3, 12) * bit(j, 4) + (1 - bit2(p4, 13)) * bit(p4, 12) * bit(j, 3) + (1 - bit2(p5, 13)) * bit(p5, 12) * bit(j, 2) + (1 - bit2(p6, 13)) * bit(p6, 12) * bit(j, 1) + (1 - bit2(p7, 13)) * bit(p7, 12) * bit(j, 0))
		&& (0 /*bh*/ == (1 - bit2(p1, 15)) * bit(p1, 14) * bit(j, 6) + (1 - bit2(p2, 15)) * bit(p2, 14) * bit(j, 5) + (1 - bit2(p3, 15)) * bit(p3, 14) * bit(j, 4) + (1 - bit2(p4, 15)) * bit(p4, 14) * bit(j, 3) + (1 - bit2(p5, 15)) * bit(p5, 14) * bit(j, 2) + (1 - bit2(p6, 15)) * bit(p6, 14) * bit(j, 1) + (1 - bit2(p7, 15)) * bit(p7, 14) * bit(j, 0))
		&& (0 /*ce*/ == (1 - bit2(p1, 17)) * bit(p1, 16) * bit(j, 6) + (1 - bit2(p2, 17)) * bit(p2, 16) * bit(j, 5) + (1 - bit2(p3, 17)) * bit(p3, 16) * bit(j, 4) + (1 - bit2(p4, 17)) * bit(p4, 16) * bit(j, 3) + (1 - bit2(p5, 17)) * bit(p5, 16) * bit(j, 2) + (1 - bit2(p6, 17)) * bit(p6, 16) * bit(j, 1) + (1 - bit2(p7, 17)) * bit(p7, 16) * bit(j, 0))
		&& (0 /*cf*/ == (1 - bit2(p1, 19)) * bit(p1, 18) * bit(j, 6) + (1 - bit2(p2, 19)) * bit(p2, 18) * bit(j, 5) + (1 - bit2(p3, 19)) * bit(p3, 18) * bit(j, 4) + (1 - bit2(p4, 19)) * bit(p4, 18) * bit(j, 3) + (1 - bit2(p5, 19)) * bit(p5, 18) * bit(j, 2) + (1 - bit2(p6, 19)) * bit(p6, 18) * bit(j, 1) + (1 - bit2(p7, 19)) * bit(p7, 18) * bit(j, 0))
		&& (0 /*cg*/ == (1 - bit2(p1, 21)) * bit(p1, 20) * bit(j, 6) + (1 - bit2(p2, 21)) * bit(p2, 20) * bit(j, 5) + (1 - bit2(p3, 21)) * bit(p3, 20) * bit(j, 4) + (1 - bit2(p4, 21)) * bit(p4, 20) * bit(j, 3) + (1 - bit2(p5, 21)) * bit(p5, 20) * bit(j, 2) + (1 - bit2(p6, 21)) * bit(p6, 20) * bit(j, 1) + (1 - bit2(p7, 21)) * bit(p7, 20) * bit(j, 0))
		&& (0 /*ch*/ == (1 - bit2(p1, 23)) * bit(p1, 22) * bit(j, 6) + (1 - bit2(p2, 23)) * bit(p2, 22) * bit(j, 5) + (1 - bit2(p3, 23)) * bit(p3, 22) * bit(j, 4) + (1 - bit2(p4, 23)) * bit(p4, 22) * bit(j, 3) + (1 - bit2(p5, 23)) * bit(p5, 22) * bit(j, 2) + (1 - bit2(p6, 23)) * bit(p6, 22) * bit(j, 1) + (1 - bit2(p7, 23)) * bit(p7, 22) * bit(j, 0))
		&& (0 /*de*/ == (1 - bit2(p1, 25)) * bit(p1, 24) * bit(j, 6) + (1 - bit2(p2, 25)) * bit(p2, 24) * bit(j, 5) + (1 - bit2(p3, 25)) * bit(p3, 24) * bit(j, 4) + (1 - bit2(p4, 25)) * bit(p4, 24) * bit(j, 3) + (1 - bit2(p5, 25)) * bit(p5, 24) * bit(j, 2) + (1 - bit2(p6, 25)) * bit(p6, 24) * bit(j, 1) + (1 - bit2(p7, 25)) * bit(p7, 24) * bit(j, 0))
		&& (0 /*df*/ == (1 - bit2(p1, 27)) * bit(p1, 26) * bit(j, 6) + (1 - bit2(p2, 27)) * bit(p2, 26) * bit(j, 5) + (1 - bit2(p3, 27)) * bit(p3, 26) * bit(j, 4) + (1 - bit2(p4, 27)) * bit(p4, 26) * bit(j, 3) + (1 - bit2(p5, 27)) * bit(p5, 26) * bit(j, 2) + (1 - bit2(p6, 27)) * bit(p6, 26) * bit(j, 1) + (1 - bit2(p7, 27)) * bit(p7, 26) * bit(j, 0))
		&& (0 /*dg*/ == (1 - bit2(p1, 29)) * bit(p1, 28) * bit(j, 6) + (1 - bit2(p2, 29)) * bit(p2, 28) * bit(j, 5) + (1 - bit2(p3, 29)) * bit(p3, 28) * bit(j, 4) + (1 - bit2(p4, 29)) * bit(p4, 28) * bit(j, 3) + (1 - bit2(p5, 29)) * bit(p5, 28) * bit(j, 2) + (1 - bit2(p6, 29)) * bit(p6, 28) * bit(j, 1) + (1 - bit2(p7, 29)) * bit(p7, 28) * bit(j, 0))
		&& (0 /*dh*/ == (1 - bit2(p1, 31)) * bit(p1, 30) * bit(j, 6) + (1 - bit2(p2, 31)) * bit(p2, 30) * bit(j, 5) + (1 - bit2(p3, 31)) * bit(p3, 30) * bit(j, 4) + (1 - bit2(p4, 31)) * bit(p4, 30) * bit(j, 3) + (1 - bit2(p5, 31)) * bit(p5, 30) * bit(j, 2) + (1 - bit2(p6, 31)) * bit(p6, 30) * bit(j, 1) + (1 - bit2(p7, 31)) * bit(p7, 30) * bit(j, 0)))
	{
		goto c2;
	}
	return;

c2: // validate c2
	for (j = 0; j < 128; j++)
	if ((0 /*ae*/ == (1 - bit2(p1, 1)) * bit(p1, 0) * bit(j, 6) + (1 - bit2(p2, 1)) * bit(p2, 0) * bit(j, 5) + (1 - bit2(p3, 1)) * bit(p3, 0) * bit(j, 4) + (1 - bit2(p4, 1)) * bit(p4, 0) * bit(j, 3) + (1 - bit2(p5, 1)) * bit(p5, 0) * bit(j, 2) + (1 - bit2(p6, 1)) * bit(p6, 0) * bit(j, 1) + (1 - bit2(p7, 1)) * bit(p7, 0) * bit(j, 0))
		&& (1 /*af*/ == (1 - bit2(p1, 3)) * bit(p1, 2) * bit(j, 6) + (1 - bit2(p2, 3)) * bit(p2, 2) * bit(j, 5) + (1 - bit2(p3, 3)) * bit(p3, 2) * bit(j, 4) + (1 - bit2(p4, 3)) * bit(p4, 2) * bit(j, 3) + (1 - bit2(p5, 3)) * bit(p5, 2) * bit(j, 2) + (1 - bit2(p6, 3)) * bit(p6, 2) * bit(j, 1) + (1 - bit2(p7, 3)) * bit(p7, 2) * bit(j, 0))
		&& (0 /*ag*/ == (1 - bit2(p1, 5)) * bit(p1, 4) * bit(j, 6) + (1 - bit2(p2, 5)) * bit(p2, 4) * bit(j, 5) + (1 - bit2(p3, 5)) * bit(p3, 4) * bit(j, 4) + (1 - bit2(p4, 5)) * bit(p4, 4) * bit(j, 3) + (1 - bit2(p5, 5)) * bit(p5, 4) * bit(j, 2) + (1 - bit2(p6, 5)) * bit(p6, 4) * bit(j, 1) + (1 - bit2(p7, 5)) * bit(p7, 4) * bit(j, 0))
		&& (0 /*ah*/ == (1 - bit2(p1, 7)) * bit(p1, 6) * bit(j, 6) + (1 - bit2(p2, 7)) * bit(p2, 6) * bit(j, 5) + (1 - bit2(p3, 7)) * bit(p3, 6) * bit(j, 4) + (1 - bit2(p4, 7)) * bit(p4, 6) * bit(j, 3) + (1 - bit2(p5, 7)) * bit(p5, 6) * bit(j, 2) + (1 - bit2(p6, 7)) * bit(p6, 6) * bit(j, 1) + (1 - bit2(p7, 7)) * bit(p7, 6) * bit(j, 0))
		&& (0 /*be*/ == (1 - bit2(p1, 9)) * bit(p1, 8) * bit(j, 6) + (1 - bit2(p2, 9)) * bit(p2, 8) * bit(j, 5) + (1 - bit2(p3, 9)) * bit(p3, 8) * bit(j, 4) + (1 - bit2(p4, 9)) * bit(p4, 8) * bit(j, 3) + (1 - bit2(p5, 9)) * bit(p5, 8) * bit(j, 2) + (1 - bit2(p6, 9)) * bit(p6, 8) * bit(j, 1) + (1 - bit2(p7, 9)) * bit(p7, 8) * bit(j, 0))
		&& (0 /*bf*/ == (1 - bit2(p1, 11)) * bit(p1, 10) * bit(j, 6) + (1 - bit2(p2, 11)) * bit(p2, 10) * bit(j, 5) + (1 - bit2(p3, 11)) * bit(p3, 10) * bit(j, 4) + (1 - bit2(p4, 11)) * bit(p4, 10) * bit(j, 3) + (1 - bit2(p5, 11)) * bit(p5, 10) * bit(j, 2) + (1 - bit2(p6, 11)) * bit(p6, 10) * bit(j, 1) + (1 - bit2(p7, 11)) * bit(p7, 10) * bit(j, 0))
		&& (0 /*bg*/ == (1 - bit2(p1, 13)) * bit(p1, 12) * bit(j, 6) + (1 - bit2(p2, 13)) * bit(p2, 12) * bit(j, 5) + (1 - bit2(p3, 13)) * bit(p3, 12) * bit(j, 4) + (1 - bit2(p4, 13)) * bit(p4, 12) * bit(j, 3) + (1 - bit2(p5, 13)) * bit(p5, 12) * bit(j, 2) + (1 - bit2(p6, 13)) * bit(p6, 12) * bit(j, 1) + (1 - bit2(p7, 13)) * bit(p7, 12) * bit(j, 0))
		&& (1 /*bh*/ == (1 - bit2(p1, 15)) * bit(p1, 14) * bit(j, 6) + (1 - bit2(p2, 15)) * bit(p2, 14) * bit(j, 5) + (1 - bit2(p3, 15)) * bit(p3, 14) * bit(j, 4) + (1 - bit2(p4, 15)) * bit(p4, 14) * bit(j, 3) + (1 - bit2(p5, 15)) * bit(p5, 14) * bit(j, 2) + (1 - bit2(p6, 15)) * bit(p6, 14) * bit(j, 1) + (1 - bit2(p7, 15)) * bit(p7, 14) * bit(j, 0))
		&& (0 /*ce*/ == (1 - bit2(p1, 17)) * bit(p1, 16) * bit(j, 6) + (1 - bit2(p2, 17)) * bit(p2, 16) * bit(j, 5) + (1 - bit2(p3, 17)) * bit(p3, 16) * bit(j, 4) + (1 - bit2(p4, 17)) * bit(p4, 16) * bit(j, 3) + (1 - bit2(p5, 17)) * bit(p5, 16) * bit(j, 2) + (1 - bit2(p6, 17)) * bit(p6, 16) * bit(j, 1) + (1 - bit2(p7, 17)) * bit(p7, 16) * bit(j, 0))
		&& (0 /*cf*/ == (1 - bit2(p1, 19)) * bit(p1, 18) * bit(j, 6) + (1 - bit2(p2, 19)) * bit(p2, 18) * bit(j, 5) + (1 - bit2(p3, 19)) * bit(p3, 18) * bit(j, 4) + (1 - bit2(p4, 19)) * bit(p4, 18) * bit(j, 3) + (1 - bit2(p5, 19)) * bit(p5, 18) * bit(j, 2) + (1 - bit2(p6, 19)) * bit(p6, 18) * bit(j, 1) + (1 - bit2(p7, 19)) * bit(p7, 18) * bit(j, 0))
		&& (0 /*cg*/ == (1 - bit2(p1, 21)) * bit(p1, 20) * bit(j, 6) + (1 - bit2(p2, 21)) * bit(p2, 20) * bit(j, 5) + (1 - bit2(p3, 21)) * bit(p3, 20) * bit(j, 4) + (1 - bit2(p4, 21)) * bit(p4, 20) * bit(j, 3) + (1 - bit2(p5, 21)) * bit(p5, 20) * bit(j, 2) + (1 - bit2(p6, 21)) * bit(p6, 20) * bit(j, 1) + (1 - bit2(p7, 21)) * bit(p7, 20) * bit(j, 0))
		&& (0 /*ch*/ == (1 - bit2(p1, 23)) * bit(p1, 22) * bit(j, 6) + (1 - bit2(p2, 23)) * bit(p2, 22) * bit(j, 5) + (1 - bit2(p3, 23)) * bit(p3, 22) * bit(j, 4) + (1 - bit2(p4, 23)) * bit(p4, 22) * bit(j, 3) + (1 - bit2(p5, 23)) * bit(p5, 22) * bit(j, 2) + (1 - bit2(p6, 23)) * bit(p6, 22) * bit(j, 1) + (1 - bit2(p7, 23)) * bit(p7, 22) * bit(j, 0))
		&& (0 /*de*/ == (1 - bit2(p1, 25)) * bit(p1, 24) * bit(j, 6) + (1 - bit2(p2, 25)) * bit(p2, 24) * bit(j, 5) + (1 - bit2(p3, 25)) * bit(p3, 24) * bit(j, 4) + (1 - bit2(p4, 25)) * bit(p4, 24) * bit(j, 3) + (1 - bit2(p5, 25)) * bit(p5, 24) * bit(j, 2) + (1 - bit2(p6, 25)) * bit(p6, 24) * bit(j, 1) + (1 - bit2(p7, 25)) * bit(p7, 24) * bit(j, 0))
		&& (0 /*df*/ == (1 - bit2(p1, 27)) * bit(p1, 26) * bit(j, 6) + (1 - bit2(p2, 27)) * bit(p2, 26) * bit(j, 5) + (1 - bit2(p3, 27)) * bit(p3, 26) * bit(j, 4) + (1 - bit2(p4, 27)) * bit(p4, 26) * bit(j, 3) + (1 - bit2(p5, 27)) * bit(p5, 26) * bit(j, 2) + (1 - bit2(p6, 27)) * bit(p6, 26) * bit(j, 1) + (1 - bit2(p7, 27)) * bit(p7, 26) * bit(j, 0))
		&& (0 /*dg*/ == (1 - bit2(p1, 29)) * bit(p1, 28) * bit(j, 6) + (1 - bit2(p2, 29)) * bit(p2, 28) * bit(j, 5) + (1 - bit2(p3, 29)) * bit(p3, 28) * bit(j, 4) + (1 - bit2(p4, 29)) * bit(p4, 28) * bit(j, 3) + (1 - bit2(p5, 29)) * bit(p5, 28) * bit(j, 2) + (1 - bit2(p6, 29)) * bit(p6, 28) * bit(j, 1) + (1 - bit2(p7, 29)) * bit(p7, 28) * bit(j, 0))
		&& (0 /*dh*/ == (1 - bit2(p1, 31)) * bit(p1, 30) * bit(j, 6) + (1 - bit2(p2, 31)) * bit(p2, 30) * bit(j, 5) + (1 - bit2(p3, 31)) * bit(p3, 30) * bit(j, 4) + (1 - bit2(p4, 31)) * bit(p4, 30) * bit(j, 3) + (1 - bit2(p5, 31)) * bit(p5, 30) * bit(j, 2) + (1 - bit2(p6, 31)) * bit(p6, 30) * bit(j, 1) + (1 - bit2(p7, 31)) * bit(p7, 30) * bit(j, 0)))
	{
		goto c3;
	}
	return;

c3: // validate c3
	for (j = 0; j < 128; j++)
	if ((0 /*ae*/ == (1 - bit2(p1, 1)) * bit(p1, 0) * bit(j, 6) + (1 - bit2(p2, 1)) * bit(p2, 0) * bit(j, 5) + (1 - bit2(p3, 1)) * bit(p3, 0) * bit(j, 4) + (1 - bit2(p4, 1)) * bit(p4, 0) * bit(j, 3) + (1 - bit2(p5, 1)) * bit(p5, 0) * bit(j, 2) + (1 - bit2(p6, 1)) * bit(p6, 0) * bit(j, 1) + (1 - bit2(p7, 1)) * bit(p7, 0) * bit(j, 0))
		&& (0 /*af*/ == (1 - bit2(p1, 3)) * bit(p1, 2) * bit(j, 6) + (1 - bit2(p2, 3)) * bit(p2, 2) * bit(j, 5) + (1 - bit2(p3, 3)) * bit(p3, 2) * bit(j, 4) + (1 - bit2(p4, 3)) * bit(p4, 2) * bit(j, 3) + (1 - bit2(p5, 3)) * bit(p5, 2) * bit(j, 2) + (1 - bit2(p6, 3)) * bit(p6, 2) * bit(j, 1) + (1 - bit2(p7, 3)) * bit(p7, 2) * bit(j, 0))
		&& (0 /*ag*/ == (1 - bit2(p1, 5)) * bit(p1, 4) * bit(j, 6) + (1 - bit2(p2, 5)) * bit(p2, 4) * bit(j, 5) + (1 - bit2(p3, 5)) * bit(p3, 4) * bit(j, 4) + (1 - bit2(p4, 5)) * bit(p4, 4) * bit(j, 3) + (1 - bit2(p5, 5)) * bit(p5, 4) * bit(j, 2) + (1 - bit2(p6, 5)) * bit(p6, 4) * bit(j, 1) + (1 - bit2(p7, 5)) * bit(p7, 4) * bit(j, 0))
		&& (0 /*ah*/ == (1 - bit2(p1, 7)) * bit(p1, 6) * bit(j, 6) + (1 - bit2(p2, 7)) * bit(p2, 6) * bit(j, 5) + (1 - bit2(p3, 7)) * bit(p3, 6) * bit(j, 4) + (1 - bit2(p4, 7)) * bit(p4, 6) * bit(j, 3) + (1 - bit2(p5, 7)) * bit(p5, 6) * bit(j, 2) + (1 - bit2(p6, 7)) * bit(p6, 6) * bit(j, 1) + (1 - bit2(p7, 7)) * bit(p7, 6) * bit(j, 0))
		&& (0 /*be*/ == (1 - bit2(p1, 9)) * bit(p1, 8) * bit(j, 6) + (1 - bit2(p2, 9)) * bit(p2, 8) * bit(j, 5) + (1 - bit2(p3, 9)) * bit(p3, 8) * bit(j, 4) + (1 - bit2(p4, 9)) * bit(p4, 8) * bit(j, 3) + (1 - bit2(p5, 9)) * bit(p5, 8) * bit(j, 2) + (1 - bit2(p6, 9)) * bit(p6, 8) * bit(j, 1) + (1 - bit2(p7, 9)) * bit(p7, 8) * bit(j, 0))
		&& (0 /*bf*/ == (1 - bit2(p1, 11)) * bit(p1, 10) * bit(j, 6) + (1 - bit2(p2, 11)) * bit(p2, 10) * bit(j, 5) + (1 - bit2(p3, 11)) * bit(p3, 10) * bit(j, 4) + (1 - bit2(p4, 11)) * bit(p4, 10) * bit(j, 3) + (1 - bit2(p5, 11)) * bit(p5, 10) * bit(j, 2) + (1 - bit2(p6, 11)) * bit(p6, 10) * bit(j, 1) + (1 - bit2(p7, 11)) * bit(p7, 10) * bit(j, 0))
		&& (0 /*bg*/ == (1 - bit2(p1, 13)) * bit(p1, 12) * bit(j, 6) + (1 - bit2(p2, 13)) * bit(p2, 12) * bit(j, 5) + (1 - bit2(p3, 13)) * bit(p3, 12) * bit(j, 4) + (1 - bit2(p4, 13)) * bit(p4, 12) * bit(j, 3) + (1 - bit2(p5, 13)) * bit(p5, 12) * bit(j, 2) + (1 - bit2(p6, 13)) * bit(p6, 12) * bit(j, 1) + (1 - bit2(p7, 13)) * bit(p7, 12) * bit(j, 0))
		&& (0 /*bh*/ == (1 - bit2(p1, 15)) * bit(p1, 14) * bit(j, 6) + (1 - bit2(p2, 15)) * bit(p2, 14) * bit(j, 5) + (1 - bit2(p3, 15)) * bit(p3, 14) * bit(j, 4) + (1 - bit2(p4, 15)) * bit(p4, 14) * bit(j, 3) + (1 - bit2(p5, 15)) * bit(p5, 14) * bit(j, 2) + (1 - bit2(p6, 15)) * bit(p6, 14) * bit(j, 1) + (1 - bit2(p7, 15)) * bit(p7, 14) * bit(j, 0))
		&& (1 /*ce*/ == (1 - bit2(p1, 17)) * bit(p1, 16) * bit(j, 6) + (1 - bit2(p2, 17)) * bit(p2, 16) * bit(j, 5) + (1 - bit2(p3, 17)) * bit(p3, 16) * bit(j, 4) + (1 - bit2(p4, 17)) * bit(p4, 16) * bit(j, 3) + (1 - bit2(p5, 17)) * bit(p5, 16) * bit(j, 2) + (1 - bit2(p6, 17)) * bit(p6, 16) * bit(j, 1) + (1 - bit2(p7, 17)) * bit(p7, 16) * bit(j, 0))
		&& (0 /*cf*/ == (1 - bit2(p1, 19)) * bit(p1, 18) * bit(j, 6) + (1 - bit2(p2, 19)) * bit(p2, 18) * bit(j, 5) + (1 - bit2(p3, 19)) * bit(p3, 18) * bit(j, 4) + (1 - bit2(p4, 19)) * bit(p4, 18) * bit(j, 3) + (1 - bit2(p5, 19)) * bit(p5, 18) * bit(j, 2) + (1 - bit2(p6, 19)) * bit(p6, 18) * bit(j, 1) + (1 - bit2(p7, 19)) * bit(p7, 18) * bit(j, 0))
		&& (0 /*cg*/ == (1 - bit2(p1, 21)) * bit(p1, 20) * bit(j, 6) + (1 - bit2(p2, 21)) * bit(p2, 20) * bit(j, 5) + (1 - bit2(p3, 21)) * bit(p3, 20) * bit(j, 4) + (1 - bit2(p4, 21)) * bit(p4, 20) * bit(j, 3) + (1 - bit2(p5, 21)) * bit(p5, 20) * bit(j, 2) + (1 - bit2(p6, 21)) * bit(p6, 20) * bit(j, 1) + (1 - bit2(p7, 21)) * bit(p7, 20) * bit(j, 0))
		&& (0 /*ch*/ == (1 - bit2(p1, 23)) * bit(p1, 22) * bit(j, 6) + (1 - bit2(p2, 23)) * bit(p2, 22) * bit(j, 5) + (1 - bit2(p3, 23)) * bit(p3, 22) * bit(j, 4) + (1 - bit2(p4, 23)) * bit(p4, 22) * bit(j, 3) + (1 - bit2(p5, 23)) * bit(p5, 22) * bit(j, 2) + (1 - bit2(p6, 23)) * bit(p6, 22) * bit(j, 1) + (1 - bit2(p7, 23)) * bit(p7, 22) * bit(j, 0))
		&& (0 /*de*/ == (1 - bit2(p1, 25)) * bit(p1, 24) * bit(j, 6) + (1 - bit2(p2, 25)) * bit(p2, 24) * bit(j, 5) + (1 - bit2(p3, 25)) * bit(p3, 24) * bit(j, 4) + (1 - bit2(p4, 25)) * bit(p4, 24) * bit(j, 3) + (1 - bit2(p5, 25)) * bit(p5, 24) * bit(j, 2) + (1 - bit2(p6, 25)) * bit(p6, 24) * bit(j, 1) + (1 - bit2(p7, 25)) * bit(p7, 24) * bit(j, 0))
		&& (0 /*df*/ == (1 - bit2(p1, 27)) * bit(p1, 26) * bit(j, 6) + (1 - bit2(p2, 27)) * bit(p2, 26) * bit(j, 5) + (1 - bit2(p3, 27)) * bit(p3, 26) * bit(j, 4) + (1 - bit2(p4, 27)) * bit(p4, 26) * bit(j, 3) + (1 - bit2(p5, 27)) * bit(p5, 26) * bit(j, 2) + (1 - bit2(p6, 27)) * bit(p6, 26) * bit(j, 1) + (1 - bit2(p7, 27)) * bit(p7, 26) * bit(j, 0))
		&& (1 /*dg*/ == (1 - bit2(p1, 29)) * bit(p1, 28) * bit(j, 6) + (1 - bit2(p2, 29)) * bit(p2, 28) * bit(j, 5) + (1 - bit2(p3, 29)) * bit(p3, 28) * bit(j, 4) + (1 - bit2(p4, 29)) * bit(p4, 28) * bit(j, 3) + (1 - bit2(p5, 29)) * bit(p5, 28) * bit(j, 2) + (1 - bit2(p6, 29)) * bit(p6, 28) * bit(j, 1) + (1 - bit2(p7, 29)) * bit(p7, 28) * bit(j, 0))
		&& (0 /*dh*/ == (1 - bit2(p1, 31)) * bit(p1, 30) * bit(j, 6) + (1 - bit2(p2, 31)) * bit(p2, 30) * bit(j, 5) + (1 - bit2(p3, 31)) * bit(p3, 30) * bit(j, 4) + (1 - bit2(p4, 31)) * bit(p4, 30) * bit(j, 3) + (1 - bit2(p5, 31)) * bit(p5, 30) * bit(j, 2) + (1 - bit2(p6, 31)) * bit(p6, 30) * bit(j, 1) + (1 - bit2(p7, 31)) * bit(p7, 30) * bit(j, 0)))
	{
		goto c4;
	}
	return;

c4: // validate c4
	for (j = 0; j < 128; j++)
	if ((0 /*ae*/ == (1 - bit2(p1, 1)) * bit(p1, 0) * bit(j, 6) + (1 - bit2(p2, 1)) * bit(p2, 0) * bit(j, 5) + (1 - bit2(p3, 1)) * bit(p3, 0) * bit(j, 4) + (1 - bit2(p4, 1)) * bit(p4, 0) * bit(j, 3) + (1 - bit2(p5, 1)) * bit(p5, 0) * bit(j, 2) + (1 - bit2(p6, 1)) * bit(p6, 0) * bit(j, 1) + (1 - bit2(p7, 1)) * bit(p7, 0) * bit(j, 0))
		&& (0 /*af*/ == (1 - bit2(p1, 3)) * bit(p1, 2) * bit(j, 6) + (1 - bit2(p2, 3)) * bit(p2, 2) * bit(j, 5) + (1 - bit2(p3, 3)) * bit(p3, 2) * bit(j, 4) + (1 - bit2(p4, 3)) * bit(p4, 2) * bit(j, 3) + (1 - bit2(p5, 3)) * bit(p5, 2) * bit(j, 2) + (1 - bit2(p6, 3)) * bit(p6, 2) * bit(j, 1) + (1 - bit2(p7, 3)) * bit(p7, 2) * bit(j, 0))
		&& (0 /*ag*/ == (1 - bit2(p1, 5)) * bit(p1, 4) * bit(j, 6) + (1 - bit2(p2, 5)) * bit(p2, 4) * bit(j, 5) + (1 - bit2(p3, 5)) * bit(p3, 4) * bit(j, 4) + (1 - bit2(p4, 5)) * bit(p4, 4) * bit(j, 3) + (1 - bit2(p5, 5)) * bit(p5, 4) * bit(j, 2) + (1 - bit2(p6, 5)) * bit(p6, 4) * bit(j, 1) + (1 - bit2(p7, 5)) * bit(p7, 4) * bit(j, 0))
		&& (0 /*ah*/ == (1 - bit2(p1, 7)) * bit(p1, 6) * bit(j, 6) + (1 - bit2(p2, 7)) * bit(p2, 6) * bit(j, 5) + (1 - bit2(p3, 7)) * bit(p3, 6) * bit(j, 4) + (1 - bit2(p4, 7)) * bit(p4, 6) * bit(j, 3) + (1 - bit2(p5, 7)) * bit(p5, 6) * bit(j, 2) + (1 - bit2(p6, 7)) * bit(p6, 6) * bit(j, 1) + (1 - bit2(p7, 7)) * bit(p7, 6) * bit(j, 0))
		&& (0 /*be*/ == (1 - bit2(p1, 9)) * bit(p1, 8) * bit(j, 6) + (1 - bit2(p2, 9)) * bit(p2, 8) * bit(j, 5) + (1 - bit2(p3, 9)) * bit(p3, 8) * bit(j, 4) + (1 - bit2(p4, 9)) * bit(p4, 8) * bit(j, 3) + (1 - bit2(p5, 9)) * bit(p5, 8) * bit(j, 2) + (1 - bit2(p6, 9)) * bit(p6, 8) * bit(j, 1) + (1 - bit2(p7, 9)) * bit(p7, 8) * bit(j, 0))
		&& (0 /*bf*/ == (1 - bit2(p1, 11)) * bit(p1, 10) * bit(j, 6) + (1 - bit2(p2, 11)) * bit(p2, 10) * bit(j, 5) + (1 - bit2(p3, 11)) * bit(p3, 10) * bit(j, 4) + (1 - bit2(p4, 11)) * bit(p4, 10) * bit(j, 3) + (1 - bit2(p5, 11)) * bit(p5, 10) * bit(j, 2) + (1 - bit2(p6, 11)) * bit(p6, 10) * bit(j, 1) + (1 - bit2(p7, 11)) * bit(p7, 10) * bit(j, 0))
		&& (0 /*bg*/ == (1 - bit2(p1, 13)) * bit(p1, 12) * bit(j, 6) + (1 - bit2(p2, 13)) * bit(p2, 12) * bit(j, 5) + (1 - bit2(p3, 13)) * bit(p3, 12) * bit(j, 4) + (1 - bit2(p4, 13)) * bit(p4, 12) * bit(j, 3) + (1 - bit2(p5, 13)) * bit(p5, 12) * bit(j, 2) + (1 - bit2(p6, 13)) * bit(p6, 12) * bit(j, 1) + (1 - bit2(p7, 13)) * bit(p7, 12) * bit(j, 0))
		&& (0 /*bh*/ == (1 - bit2(p1, 15)) * bit(p1, 14) * bit(j, 6) + (1 - bit2(p2, 15)) * bit(p2, 14) * bit(j, 5) + (1 - bit2(p3, 15)) * bit(p3, 14) * bit(j, 4) + (1 - bit2(p4, 15)) * bit(p4, 14) * bit(j, 3) + (1 - bit2(p5, 15)) * bit(p5, 14) * bit(j, 2) + (1 - bit2(p6, 15)) * bit(p6, 14) * bit(j, 1) + (1 - bit2(p7, 15)) * bit(p7, 14) * bit(j, 0))
		&& (0 /*ce*/ == (1 - bit2(p1, 17)) * bit(p1, 16) * bit(j, 6) + (1 - bit2(p2, 17)) * bit(p2, 16) * bit(j, 5) + (1 - bit2(p3, 17)) * bit(p3, 16) * bit(j, 4) + (1 - bit2(p4, 17)) * bit(p4, 16) * bit(j, 3) + (1 - bit2(p5, 17)) * bit(p5, 16) * bit(j, 2) + (1 - bit2(p6, 17)) * bit(p6, 16) * bit(j, 1) + (1 - bit2(p7, 17)) * bit(p7, 16) * bit(j, 0))
		&& (1 /*cf*/ == (1 - bit2(p1, 19)) * bit(p1, 18) * bit(j, 6) + (1 - bit2(p2, 19)) * bit(p2, 18) * bit(j, 5) + (1 - bit2(p3, 19)) * bit(p3, 18) * bit(j, 4) + (1 - bit2(p4, 19)) * bit(p4, 18) * bit(j, 3) + (1 - bit2(p5, 19)) * bit(p5, 18) * bit(j, 2) + (1 - bit2(p6, 19)) * bit(p6, 18) * bit(j, 1) + (1 - bit2(p7, 19)) * bit(p7, 18) * bit(j, 0))
		&& (0 /*cg*/ == (1 - bit2(p1, 21)) * bit(p1, 20) * bit(j, 6) + (1 - bit2(p2, 21)) * bit(p2, 20) * bit(j, 5) + (1 - bit2(p3, 21)) * bit(p3, 20) * bit(j, 4) + (1 - bit2(p4, 21)) * bit(p4, 20) * bit(j, 3) + (1 - bit2(p5, 21)) * bit(p5, 20) * bit(j, 2) + (1 - bit2(p6, 21)) * bit(p6, 20) * bit(j, 1) + (1 - bit2(p7, 21)) * bit(p7, 20) * bit(j, 0))
		&& (0 /*ch*/ == (1 - bit2(p1, 23)) * bit(p1, 22) * bit(j, 6) + (1 - bit2(p2, 23)) * bit(p2, 22) * bit(j, 5) + (1 - bit2(p3, 23)) * bit(p3, 22) * bit(j, 4) + (1 - bit2(p4, 23)) * bit(p4, 22) * bit(j, 3) + (1 - bit2(p5, 23)) * bit(p5, 22) * bit(j, 2) + (1 - bit2(p6, 23)) * bit(p6, 22) * bit(j, 1) + (1 - bit2(p7, 23)) * bit(p7, 22) * bit(j, 0))
		&& (0 /*de*/ == (1 - bit2(p1, 25)) * bit(p1, 24) * bit(j, 6) + (1 - bit2(p2, 25)) * bit(p2, 24) * bit(j, 5) + (1 - bit2(p3, 25)) * bit(p3, 24) * bit(j, 4) + (1 - bit2(p4, 25)) * bit(p4, 24) * bit(j, 3) + (1 - bit2(p5, 25)) * bit(p5, 24) * bit(j, 2) + (1 - bit2(p6, 25)) * bit(p6, 24) * bit(j, 1) + (1 - bit2(p7, 25)) * bit(p7, 24) * bit(j, 0))
		&& (0 /*df*/ == (1 - bit2(p1, 27)) * bit(p1, 26) * bit(j, 6) + (1 - bit2(p2, 27)) * bit(p2, 26) * bit(j, 5) + (1 - bit2(p3, 27)) * bit(p3, 26) * bit(j, 4) + (1 - bit2(p4, 27)) * bit(p4, 26) * bit(j, 3) + (1 - bit2(p5, 27)) * bit(p5, 26) * bit(j, 2) + (1 - bit2(p6, 27)) * bit(p6, 26) * bit(j, 1) + (1 - bit2(p7, 27)) * bit(p7, 26) * bit(j, 0))
		&& (0 /*dg*/ == (1 - bit2(p1, 29)) * bit(p1, 28) * bit(j, 6) + (1 - bit2(p2, 29)) * bit(p2, 28) * bit(j, 5) + (1 - bit2(p3, 29)) * bit(p3, 28) * bit(j, 4) + (1 - bit2(p4, 29)) * bit(p4, 28) * bit(j, 3) + (1 - bit2(p5, 29)) * bit(p5, 28) * bit(j, 2) + (1 - bit2(p6, 29)) * bit(p6, 28) * bit(j, 1) + (1 - bit2(p7, 29)) * bit(p7, 28) * bit(j, 0))
		&& (1 /*dh*/ == (1 - bit2(p1, 31)) * bit(p1, 30) * bit(j, 6) + (1 - bit2(p2, 31)) * bit(p2, 30) * bit(j, 5) + (1 - bit2(p3, 31)) * bit(p3, 30) * bit(j, 4) + (1 - bit2(p4, 31)) * bit(p4, 30) * bit(j, 3) + (1 - bit2(p5, 31)) * bit(p5, 30) * bit(j, 2) + (1 - bit2(p6, 31)) * bit(p6, 30) * bit(j, 1) + (1 - bit2(p7, 31)) * bit(p7, 30) * bit(j, 0)))
	{
		goto sln;
	}
	return;

	sln: // solution found
	*res = true;
}

int main()
{
	cudaError_t cudaStatus;
	unsigned int* dev_p = 0;
	uint3** dev_t = 0;
	bool* dev_r = 0;
	bool r = false;

	cudaStatus = initialize(&dev_p, &dev_t, &dev_r);
	if (cudaStatus != cudaSuccess) {
		goto CLEANUP;
	}

	printf("START:\n");
	uint3 n = make_uint3(0, 0, 0);

	info("n = (%d, %d, %d): ", n.x, n.y, n.z);
	calculate <<<1, 1>>>(dev_p, dev_t, n, dev_r);

	cudaStatus = cudaMemcpy(&r, dev_r, sizeof(bool), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto CLEANUP;
	}

	printf("%s\n", r ? "TRUE" : "FALSE");

CLEANUP:
	cudaFree(dev_p);

	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	return 0;
}