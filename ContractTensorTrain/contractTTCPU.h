#ifndef CONTRACTTTCPU_H
#define CONTRACTTTCPU_H

#include "contractTTGPU.h"

typedef struct _tensorSizeCPU
{
	int dim;
	int* size;
	int dataSize;
	double* hostData;
} sTensorCPU;


sTensorCPU randomTensorCPU(int dim, int* size,int mod);
sTensorCPU randomTensorCPU(int s1, int s2, int s3,int mod);
void freeTensor(sTensorCPU tensor);
void printTensor(sTensorCPU &tensor);

sTensorCPU contractTensor(sTensorCPU &tensorIn1, sTensorCPU &tensorIn2, int ind1, int ind2);
sTensorCPU contractTensor(sTensorCPU &tensorIn1, sTensorCPU &tensorIn2);

double compareTensor(sTensorGPU &tensorGPU, sTensorCPU &tensorCPU);

#endif