// Utilities and system includes
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>

// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <typedef.h>

#include<contractTTCPU.h>
#include<contractTTGPU.h>


extern "C" void contractTensorPerm(type *A, type *B, type* C, int sizeA1, int sizeA2, int sizeA3, int sizeB1, int sizeB2, int sizeB3);
extern "C" void contractTensorStart(type *A, type *B, type* C, int sizeA1, int sizeA2, int sizeA3, int sizeB1, int sizeB2, int sizeB3);
extern "C" void contractTensorFin1(type *A, type *B, type* C, int sizeA1, int sizeA2, int sizeA3, int sizeA4, int sizeB1, int sizeB2, int sizeB3);
extern "C" void contractTensorFin2(type *A, type *B, type* C, int size);

void handleCudaError(cudaError_t error){
    if (error != cudaSuccess)
    {
        printf("error code %d\n", error);
    }
}

sTensorGPU randomTensorGPU(int dim,int* size,int mod)
{
    sTensorGPU tensor;
    tensor.dim = dim;
    tensor.size = size;
    int dataSize = 1;
    for(int i=0;i<dim;i++)
        dataSize *= size[i];
    tensor.dataSize = dataSize;
    int memSize = sizeof(type)*dataSize;
    type* hostData = (type*)malloc(memSize);
    type* deviceData;
    for (int i = 0; i < tensor.dataSize; ++i){
        //type r = ((type)rand() / (type)RAND_MAX -0.5)*2;
        //hostData[i] = r;
		hostData[i] = (type)(rand() % mod + 1);
    }
    handleCudaError(cudaMalloc((void **) &deviceData, memSize));
    handleCudaError(cudaMemcpy(deviceData, hostData, memSize, cudaMemcpyHostToDevice));
    tensor.deviceData = deviceData;

    free(hostData);

    return tensor;
}

sTensorGPU randomTensorGPU(int s1,int s2,int mod){
    int* size = (int*)malloc(sizeof(int)*2);
    size[0]=s1;
    size[1]=s2;
    return randomTensorGPU(2,size,mod);
}

sTensorGPU randomTensorGPU(int s1,int s2,int s3,int mod){
    int* size = (int*)malloc(sizeof(int)*3);
    size[0]=s1;
    size[1]=s2;
    size[2]=s3;
    return randomTensorGPU(3,size,mod);
}

sTensorGPU randomTensorGPU(int s1, int s2, int s3,int s4, int mod){
	int* size = (int*)malloc(sizeof(int) * 4);
	size[0] = s1;
	size[1] = s2;
	size[2] = s3;
	size[3] = s4;
	return randomTensorGPU(4, size, mod);
}
void freeTensor(sTensorGPU tensor){
    cudaFree(tensor.deviceData);
    free(tensor.size);
}

void printTensor(sTensorGPU &tensor, int size){
    printf("Size=[");
    for(int i=0;i<tensor.dim;i++){
        printf("%d ",tensor.size[i]);
    }
    printf("](%d)\n",tensor.dataSize);
    type* out = (type*) malloc(sizeof(type)*tensor.dataSize);
    cudaMemcpy(out, tensor.deviceData, sizeof(type)*tensor.dataSize, cudaMemcpyDeviceToHost);
    for(int i=0;i<tensor.dataSize&&i<size;i++){
        //int z = (int) out[i];
        //printf("%d ",z);
        printf("%.0f ",out[i]);
		if (i % 32 == 31)
			printf("\n");
    }
    free(out);
    printf("\n");

}
#define NUM_ITER 5
sTensorGPU contractTensorPerm(sTensorGPU &tensorIn1, sTensorGPU &tensorIn2)
{
    sTensorGPU tensorOut;
    tensorOut.dim = tensorIn1.dim + tensorIn2.dim - 2;
    tensorOut.size = (int*) malloc(sizeof(int)*tensorOut.dim);
    int sizeA1 = tensorIn1.size[0];
    int sizeA2 = tensorIn1.size[1];
    int sizeA3 = tensorIn1.size[2];
    int sizeB1 = tensorIn2.size[0];
    int sizeB2 = tensorIn2.size[1];
    int sizeB3 = tensorIn2.size[2];

	int dataSize = sizeA2*sizeA3*sizeB2*sizeB3;
    if(sizeA1!=sizeB1)
        printf("Unequal Size %d!=%d\n", sizeA1, sizeB1);

	tensorOut.size[0] = sizeA2;
	tensorOut.size[1] = sizeB2;
	tensorOut.size[2] = sizeB3;
	tensorOut.size[3] = sizeA3;

    tensorOut.dataSize = dataSize;
    type* deviceData;
    handleCudaError( cudaMalloc((void **) &deviceData, sizeof(type)*tensorOut.dataSize));
	tensorOut.deviceData = deviceData;

	for (int i1 = 0; i1 < NUM_ITER; i1++){
		cudaEvent_t start;
		cudaEventCreate(&start);
		cudaEvent_t stop;
		cudaEventCreate(&stop);
		cudaEventRecord(start, NULL);

		contractTensorPerm(tensorIn1.deviceData, tensorIn2.deviceData,tensorOut.deviceData, sizeA1, sizeA2, sizeA3, sizeB1, sizeB2, sizeB3);

		cudaEventRecord(stop, NULL);
		cudaEventSynchronize(stop);
		float msecTotal = 0.0f;
		double flops = 2.0 * sizeA1 * sizeA2 * sizeA3 * sizeB2 * sizeB3;
		cudaEventElapsedTime(&msecTotal, start, stop);
		double gigaFlops = (flops * 1.0e-9f) / (msecTotal / 1000.0f);
		//printf("Time: %.3fms\n",msecTotal);
		//printf("Ops: %.0f\n", flops);
		printf("Perf= %.2f GFlop/s\n", gigaFlops);
	}

    freeTensor(tensorIn1);
    freeTensor(tensorIn2);
    return tensorOut;
}
sTensorGPU contractTensorStart(sTensorGPU &tensorIn1, sTensorGPU &tensorIn2){

	sTensorGPU tensorOut;
	tensorOut.dim = tensorIn1.dim + tensorIn2.dim - 2;
	tensorOut.size = (int*)malloc(sizeof(int)*tensorOut.dim);
	int sizeA1 = tensorIn1.size[0];
	int sizeA2 = tensorIn1.size[1];
	int sizeA3 = tensorIn1.size[2];
	int sizeB1 = tensorIn2.size[0];
	int sizeB2 = tensorIn2.size[1];
	int sizeB3 = tensorIn2.size[2];

	int dataSize = sizeA1*sizeA3*sizeB1*sizeB3;
	if (sizeA2 != sizeB2)
		printf("Unequal Size %d!=%d\n", sizeA2, sizeB2);

	tensorOut.size[0] = sizeA3;
	tensorOut.size[1] = sizeB3;
	tensorOut.size[2] = sizeA1;
	tensorOut.size[3] = sizeB1;

	tensorOut.dataSize = dataSize;
	type* deviceData;
	handleCudaError(cudaMalloc((void **)&deviceData, sizeof(type)*tensorOut.dataSize));
	tensorOut.deviceData = deviceData;

	for (int i1 = 0; i1 < NUM_ITER; i1++){
		cudaEvent_t start;
		cudaEventCreate(&start);
		cudaEvent_t stop;
		cudaEventCreate(&stop);
		cudaEventRecord(start, NULL);

		contractTensorStart(tensorIn1.deviceData, tensorIn2.deviceData,tensorOut.deviceData, sizeA1, sizeA2, sizeA3, sizeB1, sizeB2, sizeB3);

		cudaEventRecord(stop, NULL);
		cudaEventSynchronize(stop);
		float msecTotal = 0.0f;
		double flops = 2.0 * sizeA1 * sizeA3 * sizeB1 * sizeB3 * sizeA2;
		cudaEventElapsedTime(&msecTotal, start, stop);
		double gigaFlops = (flops * 1.0e-9f) / (msecTotal / 1000.0f);
		//printf("Time: %.3fms\n",msecTotal);
		//printf("Ops: %.0f\n", flops);
		printf("Perf= %.2f GFlop/s\n", gigaFlops);
	}

	freeTensor(tensorIn1);
	freeTensor(tensorIn2);
	return tensorOut;
}
sTensorGPU contractTensorFin1(sTensorGPU &tensorIn1, sTensorGPU &tensorIn2){

	sTensorGPU tensorOut;
	tensorOut.dim = 3;
	tensorOut.size = (int*)malloc(sizeof(int)*tensorOut.dim);
	int sizeA1 = tensorIn1.size[0];
	int sizeA2 = tensorIn1.size[1];
	int sizeA3 = tensorIn1.size[2];
	int sizeA4 = tensorIn1.size[3];
	int sizeB1 = tensorIn2.size[0];
	int sizeB2 = tensorIn2.size[1];
	int sizeB3 = tensorIn2.size[2];

	int dataSize = sizeA2*sizeA4*sizeB2;
	if (sizeA1 != sizeB1)
		printf("Unequal Size(0) %d!=%d\n", sizeA1, sizeB1);
	if (sizeA3 != sizeB3)
		printf("Unequal Size(2) %d!=%d\n", sizeA3, sizeB3);

	tensorOut.size[0] = sizeA2;
	tensorOut.size[1] = sizeB2;
	tensorOut.size[2] = sizeA4;

	tensorOut.dataSize = dataSize;
	type* deviceData;
	handleCudaError(cudaMalloc((void **)&deviceData, sizeof(type)*tensorOut.dataSize));
	tensorOut.deviceData = deviceData;

	for (int i1 = 0; i1 < NUM_ITER; i1++){
		cudaEvent_t start;
		cudaEventCreate(&start);
		cudaEvent_t stop;
		cudaEventCreate(&stop);
		cudaEventRecord(start, NULL);

		contractTensorFin1(tensorIn1.deviceData, tensorIn2.deviceData, tensorOut.deviceData, sizeA1, sizeA2, sizeA3, sizeA4,sizeB1, sizeB2, sizeB3);

		cudaEventRecord(stop, NULL);
		cudaEventSynchronize(stop);
		float msecTotal = 0.0f;
		double flops = 2.0 * sizeA1 * sizeA3 * sizeB1 * sizeB3 * sizeA2;
		cudaEventElapsedTime(&msecTotal, start, stop);
		double gigaFlops = (flops * 1.0e-9f) / (msecTotal / 1000.0f);
		//printf("Time: %.3fms\n",msecTotal);
		//printf("Ops: %.0f\n", flops);
		printf("Perf= %.2f GFlop/s\n", gigaFlops);
	}

	freeTensor(tensorIn1);
	freeTensor(tensorIn2);
	return tensorOut;
}
sTensorGPU contractTensorFin2(sTensorGPU &tensorIn1, sTensorGPU &tensorIn2){

	sTensorGPU tensorOut;
	tensorOut.dim = 0;
	tensorOut.size = NULL;
	int sizeA1 = tensorIn1.size[0];
	int sizeA2 = tensorIn1.size[1];
	int sizeA3 = tensorIn1.size[2];
	int sizeB1 = tensorIn2.size[0];
	int sizeB2 = tensorIn2.size[1];
	int sizeB3 = tensorIn2.size[2];
	int size = sizeA1 * sizeA2 * sizeA3;

	int dataSize = 1;
	if (sizeA1 != sizeB1)
		printf("Unequal Size(0) %d!=%d\n", sizeA1, sizeB1);
	if (sizeA2 != sizeB2)
		printf("Unequal Size(1) %d!=%d\n", sizeA2, sizeB2);
	if (sizeA3 != sizeB3)
		printf("Unequal Size(2) %d!=%d\n", sizeA3, sizeB3);

	tensorOut.dataSize = dataSize;
	type* deviceData;
	handleCudaError(cudaMalloc((void **)&deviceData, sizeof(type)*tensorOut.dataSize));
	tensorOut.deviceData = deviceData;

	for (int i1 = 0; i1 < NUM_ITER; i1++){

		cudaEvent_t start;
		cudaEventCreate(&start);
		cudaEvent_t stop;
		cudaEventCreate(&stop);
		cudaEventRecord(start, NULL);

		contractTensorFin2(tensorIn1.deviceData, tensorIn2.deviceData, tensorOut.deviceData, size);
		//cublasDdot(handle, size, tensorIn1.deviceData, 0, tensorIn2.deviceData, 0, tensorOut.deviceData);

		cudaEventRecord(stop, NULL);
		cudaEventSynchronize(stop);
		float msecTotal = 0.0f;
		double flops = 2.0 * sizeA1 * sizeA2 * sizeA3;
		cudaEventElapsedTime(&msecTotal, start, stop);
		double gigaFlops = (flops * 1.0e-9f) / (msecTotal / 1000.0f);
		//printf("Time: %.3fms\n",msecTotal);
		//printf("Ops: %.0f\n", flops);
		printf("Perf= %.2f GFlop/s\n", gigaFlops);

	}

	freeTensor(tensorIn1);
	freeTensor(tensorIn2);
	return tensorOut;
}

int main(int argc, char **argv)
{

    int size=32;
    int randSeed = 1;
#define compare	
#ifdef compare
	srand(randSeed);
	sTensorCPU t1C = randomTensorCPU(size,2,size,5);
    sTensorCPU t2C = randomTensorCPU(size,2,size,5);
//    printTensor(t1C,32);
//    printTensor(t2C,32);
    
    sTensorCPU outC = contractTensorFin2(t1C,t2C);
	printTensor(outC,32);
#endif

    srand(randSeed);
    sTensorGPU t1G = randomTensorGPU(size,2,size,5);
    sTensorGPU t2G = randomTensorGPU(size,2,size,5);
	sTensorGPU outG = contractTensorFin2(t1G, t2G);

#ifdef compare
	printTensor(outG,32);
    printf("\nDiff: %f\n",compareTensor(outG,outC));
#endif

	freeTensor(outG);

	printf("done\n");
	cudaDeviceReset();
    exit(EXIT_SUCCESS);
}
