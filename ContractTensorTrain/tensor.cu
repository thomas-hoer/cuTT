// System includes
#include <stdio.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include <device_launch_parameters.h>

__device__ void daxpy(double a, double *b, double *c)
{
	c[0] += a*b[0];
	c[1] += a*b[1];
	c[2] += a*b[2];
	c[3] += a*b[3];
	c[4] += a*b[4];
	c[5] += a*b[5];
	c[6] += a*b[6];
	c[7] += a*b[7];
	c[8] += a*b[8];
	c[9] += a*b[9];
	c[10] += a*b[10];
	c[11] += a*b[11];
	c[12] += a*b[12];
	c[13] += a*b[13];
	c[14] += a*b[14];
	c[15] += a*b[15];
}
/*
 *    +----------------+
 *    |                |
 *    |A1  |A2  |A3    |B1  |B2  =>    |A2  |B2  |A3
 *   ###############  ##########      ################
 */
__global__ void contractTensorPerm(double *A, double *B, double* C, int sizeA2, int sizeA3, int sizeB2, int contract)
{
	const int idx = threadIdx.x;
	const int idy = threadIdx.y*16;

	const int inx = threadIdx.x + blockIdx.x * blockDim.x;
	const int iny = (threadIdx.y + blockIdx.y * blockDim.y) * 16;
	const int inz = threadIdx.z + blockIdx.z * blockDim.z;

	double sum[16] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	A += inx + inz * contract * sizeA2;
	B += iny * contract;
	C += inx + iny*sizeA2 + inz * sizeA2 * sizeB2;
	__shared__ double As[32][32];
	for (int i = 0; i < contract; i += 32){
#pragma unroll
		for (int k = 0; k < 16; k ++){
			As[idx][idy + k] = A[(k + idy)*contract];
		}
		__syncthreads();
		for (int k = 0; k < 32; k++){
			//double a = A[k];
			double a = As[k][idx];
#pragma unroll
			for (int j = 0; j < 16; j++){
				sum[j] += a * B[contract *  j];
			}
			B++;
		}
		A += 32;
		__syncthreads();
	}
#pragma unroll
	for (int j = 0; j < 16; j++)
		C[j * sizeA2] = sum[j];

}
__global__ void contractTensorPermBackup(double *A, double *B, double* C, int sizeA2, int sizeA3, int sizeB2, int contract)
{
	const int idx = threadIdx.x;
	const int idy = threadIdx.y;
	const int inx = (threadIdx.x + blockIdx.x * blockDim.x) * 16;
	const int iny = threadIdx.y + blockIdx.y * blockDim.y;
	const int inz = threadIdx.z + blockIdx.z * blockDim.z;

	double sum[16] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	__shared__ double As[32][2][16];
	A += inx * contract + inz * contract * sizeA2;
	B += contract*iny;
	C += inx + iny * sizeA2 + inz * sizeA2 * sizeB2;
	for (int i = 0; i < contract; i += 32){
#pragma unroll
		for (int k = 0; k < 16; k++)
			As[idy][idx][k] = A[k * contract + idy];
		__syncthreads();
#pragma unroll
		for (int k = 0; k < 32; k++){
			daxpy(B[k], &As[k][idx][0], sum);
		}
		B += 32;
		A += 32;
		__syncthreads();
	}
#pragma unroll
	for (int j = 0; j < 16; j++){
		C[j] = sum[j];
	}

}
__global__ void contractTensorPermBackup2(double *A, double *B, double* C, int sizeA2, int sizeA3, int sizeB2, int contract)
{
	const int idx = threadIdx.x;
	const int idy = threadIdx.y * 16;

	const int inx = threadIdx.x + blockIdx.x * blockDim.x;
	const int iny = (threadIdx.y + blockIdx.y * blockDim.y) * 16;
	const int inz = threadIdx.z + blockIdx.z * blockDim.z;

	double sum[16] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	A += inx * contract + inz * contract * sizeA2;
	B += iny * contract;
	C += inx + iny*sizeA2 + inz * sizeA2 * sizeB2;
		__shared__ double Bs[32][32];
	for (int i = 0; i < contract; i += 32){
#pragma unroll
		for (int k = 0; k < 16; k ++){
		Bs[idy + k][idx] = B[contract*k+idx];
		}
		__syncthreads();
		for (int k = 0; k < 32; k++){
			double a = A[k];
#pragma unroll
			for (int j = 0; j < 16; j++){
				sum[j] += a* Bs[j + idy][k];
				//sum[j] += a * B[contract *  j];
			}
			B++;
		}
		A += 32;
		__syncthreads();
	}
#pragma unroll
	for (int j = 0; j < 16; j++)
		C[j * sizeA2] = sum[j];

}
__global__ void contractTensorPermBackup3(double *A, double *B, double* C, int sizeA2, int sizeA3, int sizeB2, int contract)
{
	const int idx = threadIdx.x;
	const int idy = threadIdx.y * 16;

	const int inx = threadIdx.x + blockIdx.x * blockDim.x;
	const int iny = (threadIdx.y + blockIdx.y * blockDim.y) * 16;
	const int inz = threadIdx.z + blockIdx.z * blockDim.z;

	double sum[16] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	A += inx * contract + inz * contract * sizeA2;
	B += iny * contract;
	C += inx + iny*sizeA2 + inz * sizeA2 * sizeB2;
	//	__shared__ double Bs[32][32];
	__shared__ double As[32][32];
	for (int i = 0; i < contract; i += 32){
		/*#pragma unroll
		for (int k = 0; k < 16; k ++){
		Bs[idy + k][idx] = B[contract*k+idx];
		}
		__syncthreads();*/
#pragma unroll
		for (int k = 0; k < 16; k++){
			As[idy + k][idx] = A[k + idy];
		}
		__syncthreads();
		for (int k = 0; k < 32; k++){
			//double a = A[k];
			double a = As[k][idx];
#pragma unroll
			for (int j = 0; j < 16; j++){
				//sum[j] += a* Bs[j + idy][k];
				sum[j] += a * B[contract *  j];
			}
			B++;
		}
		A += 32;
		__syncthreads();
	}
#pragma unroll
	for (int j = 0; j < 16; j++)
		C[j * sizeA2] = sum[j];

}

extern "C" void contractTensor(double *A, double *B, double* C, int sizeA1, int sizeA2, int sizeA3, int sizeB1, int sizeB2, int sizeB3, int indA, int indB){
	sizeB2 *= sizeB3;
	
	dim3 threads(32, 2, 1);
	dim3 grid(sizeA2 / threads.x, sizeB2 / threads.y/16, sizeA3 / threads.z);
	contractTensorPerm <<<grid, threads >>>(A, B, C, sizeA2, sizeA3, sizeB2, sizeA1);
	
}
