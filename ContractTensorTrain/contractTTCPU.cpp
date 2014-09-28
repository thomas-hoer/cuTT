#include <typedef.h>

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
	type* hostData;
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
	int memSize = sizeof(type)*dataSize;
	type* hostData = (type*)malloc(memSize);
	for (int i = 0; i < tensor.dataSize; ++i){
		//type r = ((type)rand() / (type)RAND_MAX -0.5)*2;
		hostData[i] = (type)(rand() % mod + 1);
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

sTensorCPU randomTensorCPU(int s1, int s2, int s3, int s4, int mod){
	int* size = (int*)malloc(sizeof(int) * 4);
	size[0] = s1;
	size[1] = s2;
	size[2] = s3;
	size[3] = s4;
	return randomTensorCPU(4, size, mod);
}

void freeTensor(sTensorCPU tensor){
	free(tensor.hostData);
	free(tensor.size);
}

void printTensor(sTensorCPU &tensor, int size){
	printf("Size=[");
	for (int i = 0; i<tensor.dim; i++){
		printf("%d ", tensor.size[i]);
	}
	printf("](%d)\n",tensor.dataSize);
	for (int i = 0; i<tensor.dataSize&&i<size; i++){
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
	tensorOut.hostData = (type*)malloc(sizeof(type)*tensorOut.dataSize);
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
					type sum = 0;
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

sTensorCPU contractTensorPerm(sTensorCPU &tensorIn1, sTensorCPU &tensorIn2)
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
	tensorOut.hostData = (type*)malloc(sizeof(type)*tensorOut.dataSize);
	int contract = tensorIn1.size[0];
	int a2 = tensorIn1.size[1];
	int a3 = tensorIn1.size[2];
	int b2 = tensorIn2.size[1] * tensorIn2.size[2];
	for (int j1 = 0; j1 < b2; j1++){
		for (int i1 = 0; i1 < a2; i1++){
			for (int i2 = 0; i2 < a3; i2++){
				type sum = 0;
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

sTensorCPU contractTensorStart(sTensorCPU &tensorIn1, sTensorCPU &tensorIn2)
{
	sTensorCPU tensorOut;
	tensorOut.dim = tensorIn1.dim + tensorIn2.dim - 2;
	tensorOut.size = (int*)malloc(sizeof(int)*tensorOut.dim);
	if (tensorIn1.size[1] != tensorIn2.size[1]){
		printf("Unequal Size %d!=%d\n", tensorIn1.size[1], tensorIn2.size[1]);
	}
	tensorOut.size[0] = tensorIn1.size[2];
	tensorOut.size[1] = tensorIn2.size[2];
	tensorOut.size[2] = tensorIn1.size[0];
	tensorOut.size[3] = tensorIn2.size[0];
	int dataSize = tensorIn1.size[2] * tensorIn2.size[2] * tensorIn1.size[0] * tensorIn2.size[0];

	tensorOut.dataSize = dataSize;
	tensorOut.hostData = (type*)malloc(sizeof(type)*tensorOut.dataSize);
	int contract = tensorIn1.size[1];
	int a1 = tensorIn1.size[0];
	int a3 = tensorIn1.size[2];
	int b1 = tensorIn2.size[0];
	int b3 = tensorIn2.size[2];
	for (int i1 = 0; i1 < a1; i1++){
		for (int i3 = 0; i3 < a3; i3++){
			for (int j1 = 0; j1 < b1; j1++){
				for (int j3 = 0; j3 < b3; j3++){
					type sum = 0;
					for (int c = 0; c < contract; c++){
						int idxIn1 = i1 + c*a1 + i3*a1*contract;
						int idxIn2 = j1 + c*b1 + j3*b1*contract;
						sum += tensorIn1.hostData[idxIn1] * tensorIn2.hostData[idxIn2];
					}
					int idxOut = i3 + j3*a3 + i1*a3*b3 + j1*a3*b3*a1;
					tensorOut.hostData[idxOut] = sum;
				}
			}
		}
	}
	freeTensor(tensorIn1);
	freeTensor(tensorIn2);
	return tensorOut;
}

sTensorCPU contractTensorFin1(sTensorCPU &tensorIn1, sTensorCPU &tensorIn2)
{
	sTensorCPU tensorOut;
	tensorOut.dim = 3;
	tensorOut.size = (int*)malloc(sizeof(int)*tensorOut.dim);
	if (tensorIn1.size[0] != tensorIn2.size[0]){
		printf("Unequal Size(0) %d!=%d\n", tensorIn1.size[0], tensorIn2.size[0]);
	}
	if (tensorIn1.size[2] != tensorIn2.size[2]){
		printf("Unequal Size(2) %d!=%d\n", tensorIn1.size[2], tensorIn2.size[2]);
	}
	tensorOut.size[0] = tensorIn1.size[1];
	tensorOut.size[1] = tensorIn2.size[1];
	tensorOut.size[2] = tensorIn1.size[3];
	int dataSize = tensorIn1.size[1] * tensorIn2.size[1] * tensorIn1.size[3];

	tensorOut.dataSize = dataSize;
	tensorOut.hostData = (type*)malloc(sizeof(type)*tensorOut.dataSize);
	int a2 = tensorIn1.size[1];
	int b2 = tensorIn2.size[1];
	int a4 = tensorIn1.size[3];
	int contract1 = tensorIn1.size[0];
	int contract3 = tensorIn1.size[2];
	for (int i2 = 0; i2 < a2; i2++){
		for (int j2 = 0; j2 < b2; j2++){
			for (int i4 = 0; i4 < a4; i4++){
				type sum = 0;
				for (int c1 = 0; c1 < contract1; c1++){
					for (int c3 = 0; c3 < contract3; c3++){
						int idxIn1 = c1 + i2 * contract1 + c3 * contract1 * a2 + i4 * contract1*a2*contract3;
						int idxIn2 = c1 + j2 * contract1 + c3 * contract1 * b2;
						sum += tensorIn1.hostData[idxIn1] * tensorIn2.hostData[idxIn2];
					}
				}
				int idxOut = i2 + j2*a2 + i4*a2*b2;
				tensorOut.hostData[idxOut] = sum;
			}
		}
	}
	freeTensor(tensorIn1);
	freeTensor(tensorIn2);
	return tensorOut;
}

sTensorCPU contractTensorFin2(sTensorCPU &tensorIn1, sTensorCPU &tensorIn2)
{
	sTensorCPU tensorOut;
	tensorOut.dim = 0;
	tensorOut.size = NULL;
	int dataSize = 1;

	tensorOut.dataSize = dataSize;
	tensorOut.hostData = (type*)malloc(sizeof(type)*tensorOut.dataSize);
	int contract = tensorIn1.size[0] * tensorIn1.size[1] * tensorIn1.size[2];
	type sum = 0;
	for (int c = 0; c < contract; c++){
		sum += tensorIn1.hostData[c] * tensorIn2.hostData[c];
	}
	tensorOut.hostData[0] = sum;
	freeTensor(tensorIn1);
	freeTensor(tensorIn2);
	return tensorOut;
}

type compareTensor(sTensorGPU &tensorGPU, sTensorCPU &tensorCPU){
	type* out = (type*)malloc(sizeof(type)*tensorGPU.dataSize);
	cudaMemcpy(out, tensorGPU.deviceData, sizeof(type)*tensorGPU.dataSize, cudaMemcpyDeviceToHost);
	type sum = 0;
	int numZero = 0;
	int numSum = 0;
	for (int i = 0; i<tensorGPU.dataSize; i++){
		type w1 = out[i];
		type w2 = tensorCPU.hostData[i];
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