// System includes
#include <stdio.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include <device_launch_parameters.h>

#include <typedef.h>

/*
*              +----------------------+
*              |                      |
*    +----------------------+         |
*    |         |            |         |
*    |A1  |A2  |A3  |A4     |B1  |B2  |B3   =>   |A2  |B2  |A4
*   ####################   ###############      ###############
*/
__global__ void contractTensorFin1(type *A, type *B, type* C, int sizeA2, int sizeA4, int sizeB2, int contract1, int contract2)
{
	const int inx = threadIdx.x + blockIdx.x * blockDim.x;
	const int iny = threadIdx.y + blockIdx.y * blockDim.y;
	const int inz = threadIdx.z + blockIdx.z * blockDim.z;

	A += inx * contract1 + inz * contract1 * sizeA2 * contract2;
	B += iny * contract1;
	C += inx + iny * sizeA2 + inz * sizeA2 * sizeB2;

	type sum = 0;

	for (int j = 0; j < contract2; j++){
		for (int i = 0; i < contract1; i++){
			sum += A[i] * B[i];
		}
		A += contract1 * sizeA2;
		B += contract1 * sizeB2;
	}
	C[0] = sum;
}

extern "C" void contractTensorFin1(type *A, type *B, type* C, int sizeA1, int sizeA2, int sizeA3,int sizeA4, int sizeB1, int sizeB2, int sizeB3, int indA, int indB){

	dim3 threads(32, 2, 1);
	dim3 grid(sizeA2 / threads.x, sizeB2 / threads.y, sizeA4 / threads.z);
	contractTensorFin1 <<<grid, threads >>>(A, B, C, sizeA2, sizeA4, sizeB2, sizeA1,sizeA3);

}
