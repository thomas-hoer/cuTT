// System includes
#include <stdio.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include <device_launch_parameters.h>

#include <typedef.h>

/*
*    +------+
*    |      |
*    |A1    |B1   =>  scalar 
*   #####  #####      
*/
__global__ void contractTensorFin2_32(type *A, type *B, type* C)
{
	const int inx = threadIdx.x * 32;
	A += inx;
	B += inx;
	type sum = 0;
	__shared__ double c[64];

	for (int i = 0; i < 32; i++){
		sum += A[i] * B[i];
	}
	c[threadIdx.x] = sum;
	__syncthreads();
	sum = 0;
	for (int i = 0; i < 64; i++)
		sum += c[i];
	if (threadIdx.x == 0)
		C[0] = sum;
}

extern "C" void contractTensorFin2(type *A, type *B, type* C, int size){

	if (size == 2048){
		dim3 threads(64, 1, 1);
		dim3 grid(2, 1, 1);
		contractTensorFin2_32 <<<grid, threads >>>(A, B, C);
	}

}
