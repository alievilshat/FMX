#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>

void info(const char* format, ...) {
	va_list arglist;
	_crt_va_start(arglist, format);
	vprintf(format, arglist);
	_crt_va_end(arglist);
}

#endif