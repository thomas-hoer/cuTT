// Utilities and system includes
#include <assert.h>
#include "helper_string.h"

// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>



typedef struct _matrixSize      // Optional Command-line multiplier for matrix sizes
{
    unsigned int uiWA, uiHA, uiWB, uiHB, uiWC, uiHC;
} sMatrixSize;

typedef struct _tensorSize
{
    int id;
	int dim;
	int* size;
	int dataSize;
    double* deviceData;
} sTensor;

void handleError(cudaError_t error){
    if (error != cudaSuccess)
    {
        printf("error code %d\n", error);
    }
}
static int ids = 0;
// Allocates a matrix with random double entries.
sTensor randomTensor(int dim,int* size)
{
    sTensor tensor;
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
		//hostData[i]= (rand()%8==0)?1:0;
        hostData[i] = rand() / (double)RAND_MAX;
    }
    handleError(cudaMalloc((void **) &deviceData, memSize));
    handleError(cudaMemcpy(deviceData, hostData, memSize, cudaMemcpyHostToDevice));
    tensor.deviceData = deviceData;
    free(hostData);

    return tensor;
}

sTensor randomTensor(int s1,int s2){
    int* size = (int*)malloc(sizeof(int)*2);
    size[0]=s1;
    size[1]=s2;
    return randomTensor(2,size);
}

sTensor randomTensor(int s1,int s2,int s3){
    int* size = (int*)malloc(sizeof(int)*3);
    size[0]=s1;
    size[1]=s2;
    size[2]=s3;
    return randomTensor(3,size);
}

void freeTensor(sTensor tensor){
    cudaFree(tensor.deviceData);
    free(tensor.size);
}

void printTensor(sTensor &tensor){
    printf("Tensor %d: Size=[",tensor.id);
    for(int i=0;i<tensor.dim;i++){
        printf("%d ",tensor.size[i]);
    }
    printf("]\n");
    double* out = (double*) malloc(sizeof(double)*tensor.dataSize);
    cudaMemcpy(out, tensor.deviceData, sizeof(double)*tensor.dataSize, cudaMemcpyDeviceToHost);
    for(int i=0;i<tensor.dataSize;i++){
        printf("%.2f ",out[i]);
    }
    free(out);
    printf("\n");

}

sTensor contractTensor(cublasHandle_t &handle, sTensor &tensorIn1, sTensor &tensorIn2, int numDim = 1)
{
    //printf("\nContract(%d) ->%d\n", numDim, tensorIn2.id);
    //printTensor(tensorIn1);
    //printTensor(tensorIn2);

    sTensor tensorOut;
    tensorOut.id = tensorIn2.id;
    tensorOut.dim = tensorIn1.dim + tensorIn2.dim - numDim*2;
    tensorOut.size = (int*) malloc(sizeof(int)*tensorOut.dim);
    int outW = 1;
    int outH = 1;
    int contract = tensorIn1.size[0];
    int test = tensorIn2.size[0];
    for(int i=1;i<numDim;i++){
        contract *= tensorIn1.size[i];
        test *= tensorIn2.size[i];
    }
    if(test!=contract){
        printf("Unequal Size %d!=%d\n", contract, test);
    }

    {
        int i=0;
        for(int j=numDim;j<tensorIn1.dim;j++){
            tensorOut.size[i++] = tensorIn1.size[j];
            outW *= tensorIn1.size[j];
        }
        for(int j=numDim;j<tensorIn2.dim;j++){
            tensorOut.size[i++] = tensorIn2.size[j];
            outH *= tensorIn2.size[j];
        }
    }
    tensorOut.dataSize = outW * outH;
    double* deviceData;
    handleError(cudaMalloc((void **) &deviceData, sizeof(double)*tensorOut.dataSize));
    tensorOut.deviceData = deviceData;

    double alpha = 1.f;
    double beta = 0.f;
    cublasStatus_t ret;
    //                                                    m     n      k                  lda x k            ldA         ldb x n          ldB                ldc x n            ldC
    ret = cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, outW, outH, contract, &alpha, tensorIn1.deviceData, contract, tensorIn2.deviceData, contract, &beta, tensorOut.deviceData, outW);
    if (ret != CUBLAS_STATUS_SUCCESS)
    {
        printf("cublasSgemm returned error code %d\n", ret);
    }

    freeTensor(tensorIn1);
    freeTensor(tensorIn2);
    return tensorOut;
}

int main(int argc, char **argv)
{
    srand(2006);

	cublasHandle_t handle;
	cublasCreate(&handle);
    const int n=25;
    int r = 1024;//rand()%450+50;
    int oldR;
    sTensor TT1[n];
    sTensor TT2[n];
    printf("Create Random TensorTrain\n");

    TT1[0]=randomTensor(2,r);
    for(int i=1;i<n-1;i++){
        oldR = r;
        //r=rand()%450+50;
        TT1[i] = randomTensor(oldR,2,r);
    }
    TT1[n-1] = randomTensor(r,2);

    //r=rand()%450+50;
    TT2[0]=randomTensor(2,r);
    for(int i=1;i<n-1;i++){
        oldR = r;
        //r=rand()%450+50;
        TT2[i] = randomTensor(oldR,2,r);
    }
    TT2[n-1] = randomTensor(r,2);

    sTensor temp;
    cudaEvent_t start;
    cudaEventCreate(&start);
    cudaEvent_t stop;
    cudaEventCreate(&stop);
    printf("Start contractTT\n");
    cudaEventRecord(start, NULL);
    temp = contractTensor(handle, TT1[0], TT2[0]);
    temp = contractTensor(handle, temp, TT1[1]);
    temp = contractTensor(handle, temp, TT2[1],2);

    for(int i=2;i<n;i++){
        temp = contractTensor(handle, temp, TT1[i]);
        temp = contractTensor(handle, temp, TT2[i], 2);
    }

    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);
    printTensor(temp);
    float msecTotal = 0.0f;
    cudaEventElapsedTime(&msecTotal, start, stop);
    printf("Time: %.3fms",msecTotal);

    cublasDestroy(handle);
    cudaDeviceReset();


    scanf_s("bla");
    exit(0);
}
