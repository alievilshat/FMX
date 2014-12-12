#ifndef UTILS_H
#define UTILS_H

#include "cuda_runtime.h"
#include <stdio.h>

void info(const char* format, ...) {
	va_list arglist;
	_crt_va_start(arglist, format);
	vprintf(format, arglist);
	_crt_va_end(arglist);
}

__device__ char uint3_cmp(uint3 a, uint3 b) {
	if (a.z > b.z) return 1;
	if (a.z < b.z) return -1;

	if (a.y > b.y) return 1;
	if (a.y < b.y) return -1;

	if (a.x > b.x) return 1;
	if (a.x < b.x) return -1;

	return 0;
}

__device__ uint3 uint3_add(uint3 a, uint3 b)
{
	uint3 res;
	asm("add.cc.u32      %0, %3, %6;\n\t"
		"addc.cc.u32     %1, %4, %7;\n\t"
		"addc.u32        %2, %5, %8;\n\t"
		: "=r"(res.x), "=r"(res.y), "=r"(res.z)
		: "r"(a.x), "r"(a.y), "r"(a.z),
		  "r"(b.x), "r"(b.y), "r"(b.z));
	return res;
}

__device__ uint3 uint3_sub(uint3 a, uint3 b)
{
	uint3 res;
	asm("sub.cc.u32      %0, %3, %6;\n\t"
		"subc.cc.u32     %1, %4, %7;\n\t"
		"subc.u32        %2, %5, %8;\n\t"
		: "=r"(res.x), "=r"(res.y), "=r"(res.z)
		: "r"(a.x), "r"(a.y), "r"(a.z),
		"r"(b.x), "r"(b.y), "r"(b.z));
	return res;
}

__device__ int bisect_left(uint3* data, int len, uint3 el) {
	int start = 0;

	while (len > 0) {

		int m = start + len/2;

		if (uint3_cmp(data[m], el) < 0) {
			len = len - (m+1 - start);
			start = m+1;
			continue;
		}

		if (start < m && uint3_cmp(data[m-1], el) >= 0) {
			len = m - start;
			continue;
		}
		
		return m;
	}
	return -1;
}

#endif