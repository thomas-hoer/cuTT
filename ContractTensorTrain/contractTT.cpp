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
#include<bigSizeTensors.h>


static double ops = 0;

extern "C" void contractTensorPerm(type *A, type *B, type* C, int sizeA1, int sizeA2, int sizeA3, int sizeB2);
extern "C" void contractTensorStart(type *A, type *B, type* C, int sizeA1, int sizeA2, int sizeA3, int sizeB1, int sizeB2, int sizeB3);
extern "C" void contractTensorFin1(type *A, type *B, type* C, int sizeA1, int sizeA2, int sizeA3, int sizeA4, int sizeB1, int sizeB2, int sizeB3);
extern "C" void contractTensorFin2(type *A, type *B, type* C, int size);

void handleCudaError(cudaError_t error){
    if (error != cudaSuccess)
    {
        printf("error code %d\n", error);
    }
}

sTensorGPU emptyTensor(int memSize, int maxDim){
	sTensorGPU tensor;
	tensor.size = (int*)malloc(sizeof(int) * maxDim);
	memSize *= sizeof(type);
	type* deviceData;
	handleCudaError(cudaMalloc((void **)&deviceData, memSize));
	tensor.deviceData = deviceData;
	return tensor;
}

sTensorGPU randomTensorGPU(int dim, int* size, int mod)
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
		type r = ((type)rand() / (type)RAND_MAX - 0.5) * 2;
		hostData[i] = r;
//		hostData[i] = (type)(rand() % mod + 1);
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
        printf("%.5e ",out[i]);
		if (i % 32 == 31)
			printf("\n");
    }
    free(out);
    printf("\n");

}
#define NUM_ITER 1
void contractTensorPerm1(sTensorGPU &tensorIn1, sTensorGPU &tensorIn2, sTensorGPU &tensorOut)
{
    //sTensorGPU tensorOut;
    tensorOut.dim = tensorIn1.dim + tensorIn2.dim - 2;
    //tensorOut.size = (int*) malloc(sizeof(int)*tensorOut.dim);
    int sizeA1 = tensorIn1.size[0];
    int sizeA2 = tensorIn1.size[1];
	int sizeA3 = tensorIn1.size[2];
	int sizeA4 = tensorIn1.size[3];
	int sizeB1 = tensorIn2.size[0];
    int sizeB2 = tensorIn2.size[1];
    int sizeB3 = tensorIn2.size[2];

	int dataSize = sizeA2*sizeA3*sizeA4*sizeB2*sizeB3;
    if(sizeA1!=sizeB1)
        printf("Unequal Size %d!=%d\n", sizeA1, sizeB1);

	tensorOut.size[0] = sizeA2;
	tensorOut.size[1] = sizeB2;
	tensorOut.size[2] = sizeB3;
	tensorOut.size[3] = sizeA3;
	tensorOut.size[4] = sizeA4;

    tensorOut.dataSize = dataSize;
    //type* deviceData;
    //handleCudaError( cudaMalloc((void **) &deviceData, sizeof(type)*tensorOut.dataSize));
	//tensorOut.deviceData = deviceData;

	//for (int i1 = 0; i1 < NUM_ITER; i1++){
		//cudaEvent_t start;
		//cudaEventCreate(&start);
		//cudaEvent_t stop;
		//cudaEventCreate(&stop);
		//cudaEventRecord(start, NULL);

		contractTensorPerm(tensorIn1.deviceData, tensorIn2.deviceData,tensorOut.deviceData, sizeA1, sizeA2, sizeA3*sizeA4, sizeB2*sizeB3);
		cudaDeviceSynchronize();

		//cudaEventRecord(stop, NULL);
		//cudaEventSynchronize(stop);
		//float msecTotal = 0.0f;
		double flops = 2.0 * sizeA1 * sizeA2 * sizeA3 *sizeA4* sizeB2 * sizeB3;
		ops += flops;
		//cudaEventElapsedTime(&msecTotal, start, stop);
		//double gigaFlops = (flops * 1.0e-9f) / (msecTotal / 1000.0f);
		//printf("Time: %.3fms\n",msecTotal);
		//printf("Ops: %.0f\n", flops);
		//printf("Perf= %.2f GFlop/s\n", gigaFlops);
	//}

    //freeTensor(tensorIn1);
    //freeTensor(tensorIn2);
    //return tensorOut;
}
void contractTensorPerm2(sTensorGPU &tensorIn1, sTensorGPU &tensorIn2, sTensorGPU &tensorOut)
{
	//sTensorGPU tensorOut;
	tensorOut.dim = tensorIn1.dim + tensorIn2.dim - 4;
	//tensorOut.size = (int*)malloc(sizeof(int)*tensorOut.dim);
	int sizeA1 = tensorIn1.size[0];
	int sizeA2 = tensorIn1.size[1];
	int sizeA3 = tensorIn1.size[2];
	int sizeA4 = tensorIn1.size[3];
	int sizeA5 = tensorIn1.size[4];
	int sizeB1 = tensorIn2.size[0];
	int sizeB2 = tensorIn2.size[1];
	int sizeB3 = tensorIn2.size[2];

	int dataSize = sizeA3*sizeA4*sizeA5*sizeB3;
	if (sizeA1 != sizeB1)
		printf("Unequal Size(1) %d!=%d\n", sizeA1, sizeB1);
	if (sizeA2 != sizeB2)
		printf("Unequal Size(2) %d!=%d\n", sizeA2, sizeB2);

	tensorOut.size[0] = sizeA3;
	tensorOut.size[1] = sizeB3;
	tensorOut.size[2] = sizeA4;
	tensorOut.size[3] = sizeA5;

	tensorOut.dataSize = dataSize;
	//type* deviceData;
	//handleCudaError(cudaMalloc((void **)&deviceData, sizeof(type)*tensorOut.dataSize));
	//tensorOut.deviceData = deviceData;

	//for (int i1 = 0; i1 < NUM_ITER; i1++){
		//cudaEvent_t start;
		//cudaEventCreate(&start);
		//cudaEvent_t stop;
		//cudaEventCreate(&stop);
		//cudaEventRecord(start, NULL);

		contractTensorPerm(tensorIn1.deviceData, tensorIn2.deviceData, tensorOut.deviceData, sizeA1*sizeA2, sizeA3, sizeA4*sizeA5, sizeB3);
		cudaDeviceSynchronize();

		//cudaEventRecord(stop, NULL);
		//cudaEventSynchronize(stop);
		//float msecTotal = 0.0f;
		double flops = 2.0 * sizeA1 * sizeA2 * sizeA3 *sizeA4*sizeA5* sizeB3;
		ops += flops;
		//cudaEventElapsedTime(&msecTotal, start, stop);
		//double gigaFlops = (flops * 1.0e-9f) / (msecTotal / 1000.0f);
		//printf("Time: %.3fms\n",msecTotal);
		//printf("Ops: %.0f\n", flops);
		//printf("Perf= %.2f GFlop/s\n", gigaFlops);
	//}

	//freeTensor(tensorIn1);
	//freeTensor(tensorIn2);
	//return tensorOut;
}
void contractTensorStart(sTensorGPU &tensorIn1, sTensorGPU &tensorIn2, sTensorGPU &tensorOut){

	//sTensorGPU tensorOut;
	tensorOut.dim = tensorIn1.dim + tensorIn2.dim - 2;
	//tensorOut.size = (int*)malloc(sizeof(int)*tensorOut.dim);
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
	//type* deviceData;
	//handleCudaError(cudaMalloc((void **)&deviceData, sizeof(type)*tensorOut.dataSize));
	//tensorOut.deviceData = deviceData;

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
		ops += flops;
		cudaEventElapsedTime(&msecTotal, start, stop);
		double gigaFlops = (flops * 1.0e-9f) / (msecTotal / 1000.0f);
		//printf("Time: %.3fms\n",msecTotal);
		//printf("Ops: %.0f\n", flops);
		//printf("Perf= %.2f GFlop/s\n", gigaFlops);
	}

	//freeTensor(tensorIn1);
	//freeTensor(tensorIn2);
	//return tensorOut;
}
void contractTensorFin1(sTensorGPU &tensorIn1, sTensorGPU &tensorIn2, sTensorGPU &tensorOut){

	//sTensorGPU tensorOut;
	tensorOut.dim = 3;
	//tensorOut.size = (int*)malloc(sizeof(int)*tensorOut.dim);
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
	//type* deviceData;
	//handleCudaError(cudaMalloc((void **)&deviceData, sizeof(type)*tensorOut.dataSize));
	//tensorOut.deviceData = deviceData;

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
		double flops = 2.0 * sizeA1 * sizeA3 * sizeB2 * sizeA2* sizeA4;
		ops += flops;
		cudaEventElapsedTime(&msecTotal, start, stop);
		double gigaFlops = (flops * 1.0e-9f) / (msecTotal / 1000.0f);
		//printf("Time: %.3fms\n",msecTotal);
		//printf("Ops: %.0f\n", flops);
		//printf("Perf= %.2f GFlop/s\n", gigaFlops);
	}

	//freeTensor(tensorIn1);
	//freeTensor(tensorIn2);
	//return tensorOut;
}
void contractTensorFin2(sTensorGPU &tensorIn1, sTensorGPU &tensorIn2, sTensorGPU &tensorOut){

	//sTensorGPU tensorOut;
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
	//type* deviceData;
	//handleCudaError(cudaMalloc((void **)&deviceData, sizeof(type)*tensorOut.dataSize));
	//tensorOut.deviceData = deviceData;

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
		ops += flops;
		cudaEventElapsedTime(&msecTotal, start, stop);
		double gigaFlops = (flops * 1.0e-9f) / (msecTotal / 1000.0f);
		//printf("Time: %.3fms\n",msecTotal);
		//printf("Ops: %.0f\n", flops);
		//printf("Perf= %.2f GFlop/s\n", gigaFlops);

	}

	//freeTensor(tensorIn1);
	//freeTensor(tensorIn2);
	//return tensorOut;
}


int main(int argc, char **argv)
{
	const int n = 5;
	const int size = 64*4;
	const int randSeed = 2002;// rand();
//#define compare	
#ifdef compare
	{
		srand(randSeed);
		sTensorCPU TT1[n];
		sTensorCPU TT2[n];
		for (int i = 0; i < n; i++){
			TT1[i] = randomTensorCPU(size, 2, size, 3);
			TT2[i] = randomTensorCPU(size, 2, size, 3);
		}
		sTensorCPU temp = contractTensorStart(TT1[0], TT2[0]);
		for (int i = 1; i < n - 1; i++){
			temp = contractTensorPerm1(temp, TT1[i]);
			temp = contractTensorPerm2(temp, TT2[i]);
		}
		temp = contractTensorFin1(temp, TT1[n - 1]);
		temp = contractTensorFin2(temp, TT2[n - 1]);
		printTensor(temp, 32);
	}
#endif
	{
		srand(randSeed);
		sTensorGPU TT1[n];
		sTensorGPU TT2[n];
		for (int i = 0; i < n; i++){
			TT1[i] = randomTensorGPU(size, 2, size, 3);
			TT2[i] = randomTensorGPU(size, 2, size, 3);
		}
		/*
		cudaEvent_t start;
		cudaEventCreate(&start);
		cudaEvent_t stop;
		cudaEventCreate(&stop);
		cudaEventRecord(start, NULL);

		sTensorGPU temp1 = emptyTensor(size*size*size*size, 4);
		sTensorGPU temp2 = emptyTensor(size*size*size*size * 2, 5);
		contractTensorStart(TT1[0], TT2[0], temp1);
		for (int i = 1; i < n - 1; i++){
			contractTensorPerm1(temp1, TT1[i], temp2);
			contractTensorPerm2(temp2, TT2[i], temp1);
		}
		contractTensorFin1(temp1, TT1[n-1], temp2);
		contractTensorFin2(temp2, TT2[n-1], temp1);

		cudaEventRecord(stop, NULL);
		cudaEventSynchronize(stop);
		float msecTotal = 0.0f;
		cudaEventElapsedTime(&msecTotal, start, stop);
		double gigaFlops = (ops * 1.0e-9f) / (msecTotal / 1000.0f);
		printf("Time: %.3fms\n",msecTotal);
		printf("Ops: %.0f\n", ops);
		printf("Perf= %.2f GFlop/s\n", gigaFlops);

		printTensor(temp1, 32);
		*/
		contractTT(TT1, TT2, n, size);
	}
	

#ifdef compare
//	printTensor(outG, 32);
//	printf("\nDiff: %f\n", compareTensor(outG, outC));
#endif

	//freeTensor(outG);

	printf("done\n");
	cudaDeviceReset();
	exit(EXIT_SUCCESS);
}
