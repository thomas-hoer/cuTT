#include <cublas_v2.h>

#include <typedef.h>

#include<contractTTGPU.h>


void contractTT(sTensorGPU *TT1, sTensorGPU *TT2, const int n, const int size);