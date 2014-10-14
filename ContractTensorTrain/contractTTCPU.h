#ifndef CONTRACTTTCPU_H
#define CONTRACTTTCPU_H

#include "contractTTGPU.h"
#include "typedef.h"

typedef struct _tensorSizeCPU
{
	int dim;
	int* size;
	int dataSize;
	type* hostData;
} sTensorCPU;

sTensorCPU copyToCPU(sTensorGPU &tensor);

sTensorCPU randomTensorCPU(int dim, int* size,int mod);
sTensorCPU randomTensorCPU(int s1, int s2, int s3, int mod);
sTensorCPU randomTensorCPU(int s1, int s2, int s3, int s4, int mod);
void freeTensor(sTensorCPU tensor);
void printTensor(sTensorCPU &tensor,int num);

sTensorCPU contractTensor(sTensorCPU &tensorIn1, sTensorCPU &tensorIn2, int ind1, int ind2);
sTensorCPU contractTensorStart(sTensorCPU &tensorIn1, sTensorCPU &tensorIn2);
sTensorCPU contractTensorPerm1(sTensorCPU &tensorIn1, sTensorCPU &tensorIn2);
sTensorCPU contractTensorPerm2(sTensorCPU &tensorIn1, sTensorCPU &tensorIn2);
sTensorCPU contractTensorFin1(sTensorCPU &tensorIn1, sTensorCPU &tensorIn2);
sTensorCPU contractTensorFin2(sTensorCPU &tensorIn1, sTensorCPU &tensorIn2);

type compareTensor(sTensorGPU &tensorGPU, sTensorCPU &tensorCPU);

#endif