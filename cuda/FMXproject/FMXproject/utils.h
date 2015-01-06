#ifndef UTILS_H
#define UTILS_H

#include "cuda_runtime.h"
#include <stdio.h>
#include <string.h>
#include <time.h>

const char* currentDateTime() {
	time_t now = time(0);
	struct tm  tstruct;
	char* buf = new char[80];
	tstruct = *localtime(&now);
	strftime(buf, 40, "%Y-%m-%d.%X", &tstruct);
	strftime(&buf[40], 40, "%Y-%m-%d", &tstruct);
	return buf;
}

void info(const char* format, ...) {
	const char* t = currentDateTime();
	
	char fname[100];
	sprintf(fname, "C:\\my\\log[%s].txt", &t[40]);
	
	FILE* f = fopen(fname, "a");

	printf("%s\t", t);
	fprintf(f, "%s\t", t);

	va_list arglist;
	_crt_va_start(arglist, format);
	vprintf(format, arglist);
	vfprintf(f, format, arglist);
	_crt_va_end(arglist);

	fclose(f);
}

__host__ __device__ char uint3_cmp(uint3 a, uint3 b) {
	if (a.z > b.z) return 1;
	if (a.z < b.z) return -1;

	if (a.y > b.y) return 1;
	if (a.y < b.y) return -1;

	if (a.x > b.x) return 1;
	if (a.x < b.x) return -1;

	return 0;
}

__host__ uint3 add(uint3 a, uint3 b) {
	unsigned long BASE = UINT_MAX;

	uint3 res;

	res.x = a.x + b.x;
	char carry = res.x < b.x ? 1 : 0;

	res.y = a.y + b.y + carry;
	carry = res.y < b.y ? 1 : 0;

	res.z = a.z + b.z + carry;
	return res;
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