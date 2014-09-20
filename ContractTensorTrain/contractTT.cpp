// Utilities and system includes
#include <assert.h>
#include "helper_string.h"

// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>


extern "C" void contractTensor(cublasHandle_t handle,const double *A, const double *B, double* C, int sizeA1, int sizeA2, int sizeA3, int sizeB1, int sizeB2, int sizeB3, int indA, int indB);


typedef struct _tensorSizeGPU
{
    int id;
	int dim;
	int* size;
	int dataSize;
    double* deviceData;
} sTensorGPU;

typedef struct _tensorSizeCPU
{
    int id;
	int dim;
	int* size;
	int dataSize;
    double* hostData;
} sTensorCPU;

static int ids = 0;

void handleError(cudaError_t error){
    if (error != cudaSuccess)
    {
        printf("error code %d\n", error);
    }
}

sTensorGPU randomTensorGPU(int dim,int* size)
{
    sTensorGPU tensor;
    tensor.id = ids++;
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
		hostData[i] = rand() % 4 + 1;
    }
    handleError(cudaMalloc((void **) &deviceData, memSize));
    handleError(cudaMemcpy(deviceData, hostData, memSize, cudaMemcpyHostToDevice));
    tensor.deviceData = deviceData;

    free(hostData);

    return tensor;
}
sTensorCPU randomTensorCPU(int dim,int* size)
{
    sTensorCPU tensor;
    tensor.id = ids++;
    tensor.dim = dim;
    tensor.size = size;
    int dataSize = 1;
    for(int i=0;i<dim;i++)
        dataSize *= size[i];
    tensor.dataSize = dataSize;
    int memSize = sizeof(double)*dataSize;
    double* hostData = (double*)malloc(memSize);
    for (int i = 0; i < tensor.dataSize; ++i){
        //double r = ((double)rand() / (double)RAND_MAX -0.5)*2;
		hostData[i] = rand() % 4 + 1;
        //hostData[i] = r;
    }
    tensor.hostData = hostData;

    return tensor;
}

sTensorGPU randomTensorGPU(int s1,int s2){
    int* size = (int*)malloc(sizeof(int)*2);
    size[0]=s1;
    size[1]=s2;
    return randomTensorGPU(2,size);
}

sTensorGPU randomTensorGPU(int s1,int s2,int s3){
    int* size = (int*)malloc(sizeof(int)*3);
    size[0]=s1;
    size[1]=s2;
    size[2]=s3;
    return randomTensorGPU(3,size);
}

sTensorCPU randomTensorCPU(int s1,int s2,int s3){
    int* size = (int*)malloc(sizeof(int)*3);
    size[0]=s1;
    size[1]=s2;
    size[2]=s3;
    return randomTensorCPU(3,size);
}

void freeTensor(sTensorGPU tensor){
    cudaFree(tensor.deviceData);
    free(tensor.size);
}
void freeTensor(sTensorCPU tensor){
    cudaFree(tensor.hostData);
    free(tensor.size);
}

double compareTensor(sTensorGPU &tensorGPU, sTensorCPU &tensorCPU){
    double* out = (double*) malloc(sizeof(double)*tensorGPU.dataSize);
    cudaMemcpy(out, tensorGPU.deviceData, sizeof(double)*tensorGPU.dataSize, cudaMemcpyDeviceToHost);
    double sum = 0;
	int numSum = 0;
    for(int i=0;i<tensorGPU.dataSize;i++){
        double w1 = out[i];
        double w2 = tensorCPU.hostData[i];
		if (w2 != w1)
			numSum++;
			//printf("%f ", w1);
		sum += abs(w1 - w2);
    }
	printf("#%d\n", numSum);
	printf("-%d\n", tensorGPU.dataSize);
    free(out);
    return sum;
}

void printTensor(sTensorGPU &tensor){
    printf("Tensor %d: Size=[",tensor.id);
    for(int i=0;i<tensor.dim;i++){
        printf("%d ",tensor.size[i]);
    }
    printf("]\n");
    double* out = (double*) malloc(sizeof(double)*tensor.dataSize);
    cudaMemcpy(out, tensor.deviceData, sizeof(double)*tensor.dataSize, cudaMemcpyDeviceToHost);
    for(int i=0;i<tensor.dataSize;i++){
        //int z = (int) out[i];
        //printf("%d ",z);
        printf("%.0f ",out[i]);
    }
    free(out);
    printf("\n");

}

void printTensor(sTensorCPU &tensor){
    printf("Tensor %d: Size=[",tensor.id);
    for(int i=0;i<tensor.dim;i++){
        printf("%d ",tensor.size[i]);
    }
    printf("]\n");
    for(int i=0;i<tensor.dataSize&&i<128;i++){
        //int z = (int) tensor.hostData[i];
        //printf("%d ",z);
        printf("%.0f ",tensor.hostData[i]);
    }
    printf("\n");

}

sTensorCPU contractTensor(sTensorCPU &tensorIn1, sTensorCPU &tensorIn2, int ind1, int ind2)
{
    sTensorCPU tensorOut;
    tensorOut.id = tensorIn2.id;
    tensorOut.dim = tensorIn1.dim + tensorIn2.dim - 2;
    tensorOut.size = (int*) malloc(sizeof(int)*tensorOut.dim);
    if(tensorIn1.size[ind1]!=tensorIn2.size[ind2]){
        printf("Unequal Size %d!=%d\n", tensorIn1.size[ind1], tensorIn2.size[ind2]);
    }
    int dataSize = 1;
    {
        int i=0;
        for(int j=0;j<tensorIn1.dim;j++)
            if(j!=ind1){
                tensorOut.size[i++]=tensorIn1.size[j];
                dataSize *= tensorIn1.size[j];
            }
        for(int j=0;j<tensorIn2.dim;j++)
            if(j!=ind2){
                tensorOut.size[i++]=tensorIn2.size[j];
                dataSize *= tensorIn2.size[j];
            }
    }
    tensorOut.dataSize = dataSize;
    tensorOut.hostData = (double*) malloc(sizeof(double)*tensorOut.dataSize);
    int contract = tensorIn1.size[ind1];
    int a1 = 1;
    int a3 = 1;
    int b1 = 1;
    int b3 = 1;
    for(int i=0;i<ind1;i++)
        a1*=tensorIn1.size[i];
    for(int i=0;i<ind2;i++)
        a3*=tensorIn2.size[i];
    for(int i=ind1+1;i<tensorIn1.dim;i++)
        b1*=tensorIn1.size[i];
    for(int i=ind2+1;i<tensorIn2.dim;i++)
        b3*=tensorIn2.size[i];
    int count = 0;
    for(int j1=0;j1<b1;j1++){
        for(int j2=0;j2<b3;j2++){
            for(int i1=0;i1<a1;i1++){
                for(int i2=0;i2<a3;i2++){
                    count++;
                    double sum = 0;
                    for(int c=0;c<contract;c++){
                        int idxIn1 = i1+c*a1+j1*a1*contract;
                        int idxIn2 = i2+c*a3+j2*a3*contract;
                        sum += tensorIn1.hostData[idxIn1] * tensorIn2.hostData[idxIn2];
                    }
                    int idxOut = i1+j1*a1+i2*a1*b1+j2*a1*b1*a3;
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
	tensorOut.id = tensorIn2.id;
	tensorOut.dim = tensorIn1.dim + tensorIn2.dim - 2;
	tensorOut.size = (int*)malloc(sizeof(int)*tensorOut.dim);
	if (tensorIn1.size[0] != tensorIn2.size[0]){
		printf("Unequal Size %d!=%d\n", tensorIn1.size[0], tensorIn2.size[0]);
	}
	int dataSize = 1;
	{
		int i = 0;
		for (int j = 1; j < tensorIn1.dim; j++){
			tensorOut.size[i++] = tensorIn1.size[j];
			dataSize *= tensorIn1.size[j];
		}

		for (int j = 1; j < tensorIn2.dim; j++){
			tensorOut.size[i++] = tensorIn2.size[j];
			dataSize *= tensorIn2.size[j];
		}
	}
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

sTensorGPU contractTensor(cublasHandle_t handle, sTensorGPU &tensorIn1, sTensorGPU &tensorIn2, int ind1, int ind2)
{
    sTensorGPU tensorOut;
    tensorOut.id = tensorIn2.id;
    tensorOut.dim = tensorIn1.dim + tensorIn2.dim - 2;
    tensorOut.size = (int*) malloc(sizeof(int)*tensorOut.dim);
    int sizeA1 = tensorIn1.size[0];
    int sizeA2 = tensorIn1.size[1];
    int sizeA3 = tensorIn1.size[2];
    int sizeB1 = tensorIn2.size[0];
    int sizeB2 = tensorIn2.size[1];
    int sizeB3 = tensorIn2.size[2];

    int contract = tensorIn1.size[ind1];
    int test = tensorIn2.size[ind2];
    int dataSize = 1;
    if(test!=contract)
        printf("Unequal Size %d!=%d\n", contract, test);
    {
        int i=0;
        for(int j=0;j<tensorIn1.dim;j++)
            if(j!=ind1){
                tensorOut.size[i++]=tensorIn1.size[j];
                dataSize *= tensorIn1.size[j];
            }
        for(int j=0;j<tensorIn2.dim;j++)
            if(j!=ind2){
                tensorOut.size[i++]=tensorIn2.size[j];
                dataSize *= tensorIn2.size[j];
            }
    }

    tensorOut.dataSize = dataSize;
    double* deviceData;
    handleError( cudaMalloc((void **) &deviceData, sizeof(double)*tensorOut.dataSize));
	tensorOut.deviceData = deviceData;

//    contractTensor(tensorIn1.deviceData, tensorIn2.deviceData, tensorOut.deviceData, sizeA1, sizeA2, sizeA3, sizeB1, sizeB2, sizeB3, 0, 0);

    cudaEvent_t start;
    cudaEventCreate(&start);
    cudaEvent_t stop;
    cudaEventCreate(&stop);
    cudaEventRecord(start, NULL);

	contractTensor(handle, tensorIn1.deviceData, tensorIn2.deviceData, tensorOut.deviceData, sizeA1, sizeA2, sizeA3, sizeB1, sizeB2, sizeB3, 0, 0);

    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);
    float msecTotal = 0.0f;
    double flops = 2.0 * sizeA1 * sizeA2 * sizeA3 * sizeB2 * sizeB3;
    cudaEventElapsedTime(&msecTotal, start, stop);
    double gigaFlops = (flops * 1.0e-9f) / (msecTotal / 1000.0f);
    printf("Time: %.3fms\n",msecTotal);
    printf("Ops: %.0f\n", flops);
    printf("Perf= %.2f GFlop/s\n", gigaFlops);

    freeTensor(tensorIn1);
    freeTensor(tensorIn2);
    return tensorOut;
}
int main(int argc, char **argv)
{

	static cublasHandle_t handle;
	cublasCreate(&handle);

    int size=64;
    int randSeed = 1000;
    srand(randSeed);
    sTensorCPU t1C = randomTensorCPU(size,size,size*size);
    sTensorCPU t2C = randomTensorCPU(size,2,size);
    //printTensor(t1C);
    //printTensor(t2C);
    
    sTensorCPU outC = contractTensor(t1C,t2C);
	//printTensor(outC);

    srand(randSeed);
    sTensorGPU t1G = randomTensorGPU(size,size,size*size);
    sTensorGPU t2G = randomTensorGPU(size,2,size);
    

	sTensorGPU outG = contractTensor(handle, t1G, t2G, 0, 0);
	//printTensor(outG);
    printf("Diff: %f",compareTensor(outG,outC));
 
	cublasDestroy(handle);
	
	scanf_s("bla");
    exit(0);
}
