// System includes
#include <stdio.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include <device_launch_parameters.h>

#include <typedef.h>

/*
 *    +----------------+
 *    |                |
 *    |A1  |A2  |A3    |B1  |B2  =>    |A2  |B2  |A3
 *   ###############  ##########      ################
 */
__global__ void contractTensorPermKernel(type *A, type *B, type* C, int sizeA2, int sizeA3, int sizeB2, int contract)
{
	const int idx = threadIdx.x;
	const int idy = threadIdx.y*16;

	const int inx = threadIdx.x + blockIdx.x * blockDim.x;
	const int iny = (threadIdx.y + blockIdx.y * blockDim.y) * 16;
	const int inz = threadIdx.z + blockIdx.z * blockDim.z;

	type sum[16] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	A += inx + inz * contract * sizeA2;
	B += iny * contract;
	C += inx + iny*sizeA2 + inz * sizeA2 * sizeB2;
	__shared__ type As[32][32];
	for (int i = 0; i < contract; i += 32){
#pragma unroll
		for (int k = 0; k < 16; k ++){
			As[idx][idy + k] = A[(k + idy)*contract];
		}
		__syncthreads();
		for (int k = 0; k < 32; k++){
			type a = As[k][idx];
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

extern "C" void contractTensorPerm(type *A, type *B, type* C, int sizeA1, int sizeA2, int sizeA3, int sizeB2){
	
	dim3 threads(32, 2, 1);
	dim3 grid(sizeA2 / threads.x, sizeB2 / threads.y/16, sizeA3 / threads.z);
	contractTensorPermKernel<<<grid, threads >>>(A, B, C, sizeA2, sizeA3, sizeB2, sizeA1);
	
}
