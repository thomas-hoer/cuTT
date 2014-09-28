#ifndef CONTRACTTTGPU_H
#define CONTRACTTTGPU_H

typedef struct _tensorSizeGPU
{
	int dim;
	int* size;
	int dataSize;
	type* deviceData;
} sTensorGPU;

#endif