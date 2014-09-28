// System includes
#include <stdio.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include <device_launch_parameters.h>

#include <typedef.h>
/*
*         +----------------+
*         |                |
*    |A1  |A2  |A3    |B1  |B2  |B3   =>    |A3  |B3  |A1  |B1
*   ###############  ###############       ####################
*/
__global__ void contractTensorStart(type *A, type *B, type* C, int sizeA1, int sizeA3, int sizeB1, int sizeB3, int contract)
{
	const int idA3 = threadIdx.x + blockIdx.x * blockDim.x * 8;
	const int idA1 = threadIdx.y + blockIdx.y * blockDim.y;
	const int idz = threadIdx.z + blockIdx.z * blockDim.z;
	const int idB1 = idz %sizeB1;
	const int idB3 = idz / sizeB1;

	A += idA1 + idA3 * sizeA1 * contract;
	int stepA = blockDim.x*sizeA1*contract;
	B += idB1 + idB3 * sizeB1 * contract;
	C += idA3 + idB3 * sizeA3 + idA1 * sizeA3 * sizeB3 + idB1 * sizeA3*sizeB3*sizeA1;

	type c[8] = { 0, 0, 0, 0,0,0,0,0 };
	type b1 = B[0];
	type b2 = B[sizeB1];
#pragma unroll
	for (int i = 0; i < 8; i++){
		c[i] = A[0] * b1;
		c[i] += A[sizeA1] * b2;
		A += stepA;
	}
	for (int i = 0; i < 8; i++){
		C[0] = c[i];
		C += blockDim.x;
	}
}

extern "C" void contractTensorStart(type *A, type *B, type* C, int sizeA1, int sizeA2, int sizeA3, int sizeB1, int sizeB2, int sizeB3){

	dim3 threads(4, 32, 1);
	dim3 grid(sizeA3 / threads.x / 8, sizeA1 / threads.y, sizeB1*sizeB3 / threads.z);
	contractTensorStart<<<grid, threads >>>(A, B, C, sizeA1, sizeA3, sizeB1, sizeB3, sizeA2);

}
