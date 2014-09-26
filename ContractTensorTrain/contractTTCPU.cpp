#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>

#include "contractTTGPU.h"
#include <cuda_runtime.h>

typedef struct _tensorSizeCPU
{
	int dim;
	int* size;
	int dataSize;
	double* hostData;
} sTensorCPU;


sTensorCPU randomTensorCPU(int dim, int* size, int mod)
{
	sTensorCPU tensor;
	tensor.dim = dim;
	tensor.size = size;
	int dataSize = 1;
	for (int i = 0; i<dim; i++)
		dataSize *= size[i];
	tensor.dataSize = dataSize;
	int memSize = sizeof(double)*dataSize;
	double* hostData = (double*)malloc(memSize);
	for (int i = 0; i < tensor.dataSize; ++i){
		//double r = ((double)rand() / (double)RAND_MAX -0.5)*2;
		hostData[i] = rand() % mod + 1;
		//hostData[i] = r;
	}
	tensor.hostData = hostData;

	return tensor;
}

sTensorCPU randomTensorCPU(int s1, int s2, int s3,int mod){
	int* size = (int*)malloc(sizeof(int) * 3);
	size[0] = s1;
	size[1] = s2;
	size[2] = s3;
	return randomTensorCPU(3, size,mod);
}

void freeTensor(sTensorCPU tensor){
	free(tensor.hostData);
	free(tensor.size);
}

void printTensor(sTensorCPU &tensor){
	printf("Size=[");
	for (int i = 0; i<tensor.dim; i++){
		printf("%d ", tensor.size[i]);
	}
	printf("](%d)\n",tensor.dataSize);
	for (int i = 0; i<tensor.dataSize&&i<64; i++){
		//int z = (int) tensor.hostData[i];
		//printf("%d ",z);
		printf("%.0f ", tensor.hostData[i]);
		if (i % 32 == 31)
			printf("\n");

	}
	printf("\n");

}

sTensorCPU contractTensor(sTensorCPU &tensorIn1, sTensorCPU &tensorIn2, int ind1, int ind2)
{
	sTensorCPU tensorOut;
	tensorOut.dim = tensorIn1.dim + tensorIn2.dim - 2;
	tensorOut.size = (int*)malloc(sizeof(int)*tensorOut.dim);
	if (tensorIn1.size[ind1] != tensorIn2.size[ind2]){
		printf("Unequal Size %d!=%d\n", tensorIn1.size[ind1], tensorIn2.size[ind2]);
	}
	int dataSize = 1;
	{
		int i = 0;
		for (int j = 0; j<tensorIn1.dim; j++)
			if (j != ind1){
			tensorOut.size[i++] = tensorIn1.size[j];
			dataSize *= tensorIn1.size[j];
			}
		for (int j = 0; j<tensorIn2.dim; j++)
			if (j != ind2){
			tensorOut.size[i++] = tensorIn2.size[j];
			dataSize *= tensorIn2.size[j];
			}
	}
	tensorOut.dataSize = dataSize;
	tensorOut.hostData = (double*)malloc(sizeof(double)*tensorOut.dataSize);
	int contract = tensorIn1.size[ind1];
	int a1 = 1;
	int a3 = 1;
	int b1 = 1;
	int b3 = 1;
	for (int i = 0; i<ind1; i++)
		a1 *= tensorIn1.size[i];
	for (int i = 0; i<ind2; i++)
		a3 *= tensorIn2.size[i];
	for (int i = ind1 + 1; i<tensorIn1.dim; i++)
		b1 *= tensorIn1.size[i];
	for (int i = ind2 + 1; i<tensorIn2.dim; i++)
		b3 *= tensorIn2.size[i];
	int count = 0;
	for (int j1 = 0; j1<b1; j1++){
		for (int j2 = 0; j2<b3; j2++){
			for (int i1 = 0; i1<a1; i1++){
				for (int i2 = 0; i2<a3; i2++){
					count++;
					double sum = 0;
					for (int c = 0; c<contract; c++){
						int idxIn1 = i1 + c*a1 + j1*a1*contract;
						int idxIn2 = i2 + c*a3 + j2*a3*contract;
						sum += tensorIn1.hostData[idxIn1] * tensorIn2.hostData[idxIn2];
					}
					int idxOut = i1 + j1*a1 + i2*a1*b1 + j2*a1*b1*a3;
					tensorOut.hostData[idxOut] = sum;
				}
			}
		}
	}
	freeTensor(tensorIn1);
	freeTensor(tensorIn2);
	return tensorOut;
}

sTensorCPU contractTensor(sTensorCPU &tensorIn1, sTensorCPU &tensorIn2)
{
	sTensorCPU tensorOut;
	tensorOut.dim = tensorIn1.dim + tensorIn2.dim - 2;
	tensorOut.size = (int*)malloc(sizeof(int)*tensorOut.dim);
	if (tensorIn1.size[0] != tensorIn2.size[0]){
		printf("Unequal Size %d!=%d\n", tensorIn1.size[0], tensorIn2.size[0]);
	}
	tensorOut.size[0] = tensorIn1.size[1];
	tensorOut.size[1] = tensorIn2.size[1];
	tensorOut.size[2] = tensorIn2.size[2];
	tensorOut.size[3] = tensorIn1.size[2];
	int dataSize = 1;
	for (int j = 1; j < tensorIn1.dim; j++)
		dataSize *= tensorIn1.size[j];
	for (int j = 1; j < tensorIn2.dim; j++)
		dataSize *= tensorIn2.size[j];
	tensorOut.dataSize = dataSize;
	tensorOut.hostData = (double*)malloc(sizeof(double)*tensorOut.dataSize);
	int contract = tensorIn1.size[0];
	int a2 = tensorIn1.size[1];
	int a3 = tensorIn1.size[2];
	int b2 = tensorIn2.size[1] * tensorIn2.size[2];
	for (int j1 = 0; j1 < b2; j1++){
		for (int i1 = 0; i1 < a2; i1++){
			for (int i2 = 0; i2 < a3; i2++){
				double sum = 0;
				for (int c = 0; c < contract; c++){
					int idxIn1 = c + i1*contract + i2*contract*a2;
					int idxIn2 = c + j1*contract;
					sum += tensorIn1.hostData[idxIn1] * tensorIn2.hostData[idxIn2];
				}
				int idxOut = i1 + j1*a2 + i2*a2*b2;
				tensorOut.hostData[idxOut] = sum;
			}
		}
	}
	freeTensor(tensorIn1);
	freeTensor(tensorIn2);
	return tensorOut;
}

double compareTensor(sTensorGPU &tensorGPU, sTensorCPU &tensorCPU){
	double* out = (double*)malloc(sizeof(double)*tensorGPU.dataSize);
	cudaMemcpy(out, tensorGPU.deviceData, sizeof(double)*tensorGPU.dataSize, cudaMemcpyDeviceToHost);
	double sum = 0;
	int numZero = 0;
	int numSum = 0;
	for (int i = 0; i<tensorGPU.dataSize; i++){
		double w1 = out[i];
		double w2 = tensorCPU.hostData[i];
		if (w2 != w1)
			numSum++;
		if (w1 == 0)
			numZero++;
		//printf("%f ", w1);
		sum += abs(w1 - w2);
	}
	printf("0-%d\n", numZero);
	printf("#%d\n", numSum);
	printf("-%d\n", tensorGPU.dataSize);
	free(out);
	return sum/tensorCPU.dataSize;
}