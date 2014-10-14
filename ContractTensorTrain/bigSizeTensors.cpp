#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>


// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <typedef.h>

#include<contractTTGPU.h>
#include<contractTTCPU.h>

static double bops = 0;
void contractTensor(cublasHandle_t &handle, sTensorGPU &tensorIn1, sTensorGPU &tensorIn2, sTensorGPU &tensorOut, int numDim = 1)
{
	//printf("\nContract(%d) ->%d\n", numDim, tensorIn2.id);
	//printTensor(tensorIn1);
	//printTensor(tensorIn2);

	//sTensor tensorOut;
	tensorOut.dim = tensorIn1.dim + tensorIn2.dim - numDim * 2;
	//tensorOut.size = (int*)malloc(sizeof(int)*tensorOut.dim);
	int outW = 1;
	int outH = 1;
	int contract = tensorIn1.size[0];
	int test = tensorIn2.size[0];
	for (int i = 1; i<numDim; i++){
		contract *= tensorIn1.size[i];
		test *= tensorIn2.size[i];
	}
	if (test != contract){
		printf("Unequal Size %d!=%d\n", contract, test);
	}

	{
		int i = 0;
		for (int j = numDim; j<tensorIn1.dim; j++){
			tensorOut.size[i++] = tensorIn1.size[j];
			outW *= tensorIn1.size[j];
		}
		for (int j = numDim; j<tensorIn2.dim; j++){
			tensorOut.size[i++] = tensorIn2.size[j];
			outH *= tensorIn2.size[j];
		}
	}
	tensorOut.dataSize = outW * outH;
	//double* deviceData;
	//handleError(cudaMalloc((void **)&deviceData, sizeof(double)*tensorOut.dataSize));
	//tensorOut.deviceData = deviceData;

	type alpha = 1.f;
	type beta = 0.f;
	cublasStatus_t ret;
	double flops = outW;
	flops *= outH;
	flops *= contract;
	flops *= 2;
	bops += flops;
	//                                                    m     n      k                  lda x k            ldA         ldb x n          ldB                ldc x n            ldC
	ret = cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, outW, outH, contract, &alpha, tensorIn1.deviceData, contract, tensorIn2.deviceData, contract, &beta, tensorOut.deviceData, outW);
	if (ret != CUBLAS_STATUS_SUCCESS)
	{
		printf("cublasSgemm returned error code %d\n", ret);
	}
	cudaDeviceSynchronize();
	//freeTensor(tensorIn1);
	//freeTensor(tensorIn2);
	//return tensorOut;
}


sTensorGPU prepareTensorStart(sTensorCPU tensorIn, int i){
	sTensorGPU tensor;
	tensor.size = (int*)malloc(sizeof(int) * 2);
	tensor.size[0] = tensorIn.size[1];
	tensor.size[1] = tensorIn.size[2];
	tensor.dim = 2;
	type* deviceData;
	tensor.dataSize = tensorIn.size[1] * tensorIn.size[2];
	cudaMalloc((void **)&deviceData, sizeof(type) * tensor.dataSize);
	type* data = (type*)malloc(sizeof(type) * tensor.dataSize);
	for (int k = 0; k < tensor.size[1]; k++){
		for (int j = 0; j < tensor.size[0]; j++){
			type value = tensorIn.hostData[i + j*tensorIn.size[0] + k*tensorIn.size[0] * tensorIn.size[1]];
			//printf("%0.f ", value);
			data[j + k*tensor.size[0]] = value;
		}
	}
	cudaMemcpy(deviceData, data, sizeof(type) * tensor.dataSize, cudaMemcpyHostToDevice);

	tensor.deviceData = deviceData;
	return tensor;
}
sTensorGPU prepareTensorEnd(sTensorCPU tensorIn, int i){
	sTensorGPU tensor;
	tensor.size = (int*)malloc(sizeof(int) * 2);
	tensor.size[0] = tensorIn.size[0];
	tensor.size[1] = tensorIn.size[1];
	tensor.dim = 2;
	type* deviceData;
	tensor.dataSize = tensorIn.size[1] * tensorIn.size[2];
	cudaMalloc((void **)&deviceData, sizeof(type) * tensor.dataSize);
	type* data = (type*)malloc(sizeof(type) * tensor.dataSize);
	for (int j = 0; j < tensor.size[0]; j++){
		for (int k = 0; k < tensor.size[1]; k++){
			data[j + k*tensor.size[0]] = tensorIn.hostData[j + k*tensorIn.size[0] + i*tensorIn.size[0] * tensorIn.size[1]];
		}
	}
	cudaMemcpy(deviceData, data, sizeof(type) *tensor.dataSize, cudaMemcpyHostToDevice);

	tensor.deviceData = deviceData;
	return tensor;
}
void contractTT(sTensorGPU *TT1, sTensorGPU *TT2, const int n, const int size)
{
	cublasHandle_t handle;
	cublasCreate(&handle);
	type result=0;

	sTensorGPU temp1 = emptyTensor(size*size,2);
	sTensorGPU temp2 = emptyTensor(size*size*2,3);
	cudaEvent_t start;
	cudaEventCreate(&start);
	cudaEvent_t stop;
	cudaEventCreate(&stop);

	//printf("Start contractTT\n");

	cudaEventRecord(start, NULL);
	int indA = TT1[0].size[0];
	int indB = TT2[0].size[0];

	sTensorCPU tt1start = copyToCPU(TT1[0]);
	sTensorCPU tt2start = copyToCPU(TT2[0]);
	sTensorCPU tt1end = copyToCPU(TT1[n - 1]);
	sTensorCPU tt2end = copyToCPU( TT2[n - 1]);


	for (int i = 0; i < indA; i++){
		TT1[0] = prepareTensorStart(tt1start, i);
		TT1[n - 1] = prepareTensorEnd(tt1end, i);
		for (int j = 0; j < indB; j++){
			TT2[0] = prepareTensorStart(tt2start, j);
			TT2[n - 1] = prepareTensorEnd(tt2end, j);
			contractTensor(handle, TT1[0], TT2[0], temp1);
			for (int i = 1; i < n; i++){
				contractTensor(handle, temp1, TT1[i], temp2);
				contractTensor(handle, temp2, TT2[i], temp1, 2);
			}
			type add = 0;
			cudaMemcpy(&add, temp1.deviceData, sizeof(type), cudaMemcpyDeviceToHost);
			//printf("%e ", add);
			result += add;
		}
	}
	cudaEventRecord(stop, NULL);
	cudaEventSynchronize(stop);
	
	float msecTotal = 0.0f;
	cudaEventElapsedTime(&msecTotal, start, stop);
	printf("Time: %.3fms\n", msecTotal);
	printf("Ops: %.0f\n", bops);
	double gigaFlops = (bops * 1.0e-9f) / (msecTotal / 1000.0f);
	printf("Perf= %.2f GFlop/s\n", gigaFlops);

	cublasDestroy(handle);
	cudaDeviceReset();

	printf("%.5e \n", result);
	exit(0);
}