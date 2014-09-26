// Utilities and system includes
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>

// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <ContractTensorTrain\contractTTCPU.h>
#include <ContractTensorTrain\contractTTGPU.h>

extern "C" void contractTensor(double *A, double *B, double* C, int sizeA1, int sizeA2, int sizeA3, int sizeB1, int sizeB2, int sizeB3, int indA, int indB);

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
    int memSize = sizeof(double)*dataSize;
    double* hostData = (double*)malloc(memSize);
    double* deviceData;
    for (int i = 0; i < tensor.dataSize; ++i){
        //double r = ((double)rand() / (double)RAND_MAX -0.5)*2;
        //hostData[i] = r;
		hostData[i] = rand() % mod + 1;
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



void freeTensor(sTensorGPU tensor){
    cudaFree(tensor.deviceData);
    free(tensor.size);
}



void printTensor(sTensorGPU &tensor){
    printf("Size=[");
    for(int i=0;i<tensor.dim;i++){
        printf("%d ",tensor.size[i]);
    }
    printf("](%d)\n",tensor.dataSize);
    double* out = (double*) malloc(sizeof(double)*tensor.dataSize);
    cudaMemcpy(out, tensor.deviceData, sizeof(double)*tensor.dataSize, cudaMemcpyDeviceToHost);
    for(int i=0;i<tensor.dataSize&&i<64;i++){
        //int z = (int) out[i];
        //printf("%d ",z);
        printf("%.0f ",out[i]);
		if (i % 32 == 31)
			printf("\n");
    }
    free(out);
    printf("\n");

}

sTensorGPU contractTensor(sTensorGPU &tensorIn1, sTensorGPU &tensorIn2)
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
    double* deviceData;
    handleCudaError( cudaMalloc((void **) &deviceData, sizeof(double)*tensorOut.dataSize));
	tensorOut.deviceData = deviceData;

	for (int i1 = 0; i1 < 5; i1++){
		cudaEvent_t start;
		cudaEventCreate(&start);
		cudaEvent_t stop;
		cudaEventCreate(&stop);
		cudaEventRecord(start, NULL);

		contractTensor(tensorIn1.deviceData, tensorIn2.deviceData, tensorOut.deviceData, sizeA1, sizeA2, sizeA3, sizeB1, sizeB2, sizeB3, 0, 0);

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
int main(int argc, char **argv)
{

    int size=32;
    int randSeed = rand();
#define compare	
#ifdef compare
	srand(randSeed);
	sTensorCPU t1C = randomTensorCPU(size,size,size*size,1000);
    sTensorCPU t2C = randomTensorCPU(size,2,size,1000);
    //printTensor(t1C);
    //printTensor(t2C);
    
    sTensorCPU outC = contractTensor(t1C,t2C);
	//printTensor(outC);
#endif

    srand(randSeed);
    sTensorGPU t1G = randomTensorGPU(size,size,size*size,1000);
    sTensorGPU t2G = randomTensorGPU(size,2,size,1000);
    

	sTensorGPU outG = contractTensor(t1G, t2G);
#ifdef compare
	//printTensor(outG);
    printf("Diff: %f\n",compareTensor(outG,outC));
#endif

	freeTensor(outG);

	printf("done");
	scanf_s("bla");
	cudaDeviceReset();
    exit(EXIT_SUCCESS);
}
