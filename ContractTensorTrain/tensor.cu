// System includes
#include <stdio.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>


const int facA = 4;
const int facB = 1;
const int facC = 16;
__global__ void contractTensor10( const double *A, const double *B, double* C, int sizeA1, int sizeA3, int sizeB3, int contract)
{
    const int inx = threadIdx.x + blockIdx.x * facA;
    const int iny = threadIdx.y + blockIdx.y * facB;
    const int inz = threadIdx.z + blockIdx.z * facC;
    const int idxC = inx + iny * sizeA1 + inz * sizeA1 * sizeA3;
    C+= idxC;
    float sum = 0;
    A += inx + iny*contract*sizeA1;
    B += contract * inz;
    for(int i = 0;i<contract;i++){
        sum += A[0] * B[0];
        A+=sizeA1;
        B++;
    }
    C[0]=sum;
}
/*
 *    +----------------+
 *    |                |
 *    |A1  |A2  |A3    |B1  |B2  =>    |A2  |B2  |A3
 *   ###############  ##########      ################
 */
__global__ void contractTensorPerm( const double *A, const double *B, double* C, int sizeA2, int sizeA3, int sizeB2, int contract)
{
    const int inx = threadIdx.x + blockIdx.x * facA;
    const int iny = threadIdx.y + blockIdx.y * facB;
    const int inz = threadIdx.z + blockIdx.z * facC;

    const int idxC = inx + inz * sizeA2 + iny * sizeA2 * sizeB2;
    double sum = 0;
    A += inx * contract + iny * contract * sizeA2;
    B += contract * inz;
#pragma unroll 16
	for(int i = 0;i<contract;i++){
        sum += A[i] * B[i];
    }
    C[idxC]=sum;
}

extern "C" void contractTensor(cublasHandle_t handle,const double *A, const double *B, double* C, int sizeA1, int sizeA2, int sizeA3, int sizeB1, int sizeB2, int sizeB3, int indA, int indB){
	sizeB2 *= sizeB3;
    dim3 grid( sizeA2/facA, sizeA3/facB, sizeB2/facC), threads(facA, facB, facC);
    contractTensorPerm<<<grid,threads>>>(A, B, C, sizeA2, sizeA3, sizeB2, sizeA1);
	
	
	double alpha = 1;
	double beta = 0;
	for (int i = 0; i < 2; i++){
		//cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, sizeB2, sizeA2, sizeA1, &alpha, A, sizeA1, B, sizeA1, &beta, C, sizeB2);

		A += sizeA1 * sizeA2;
		C += sizeA2 * sizeB2;
	}

}