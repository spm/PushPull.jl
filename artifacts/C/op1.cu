#define CUDA
#include "cuheader.h"
#include "op1_dev.cu"
/*
    MAXD - maximum filter size
    MAXN - maximum number of gradient fields
*/

__global__ void op1(float *u, const USIZE_t *d, const float *g, const float *a, const float *b, const float *s, const USIZE_t *o, const USIZE_t *n)
{
    USIZE_t ijk, i, j, k;
    ijk = threadIdx.x + blockDim.x*blockIdx.x;
    i   = (ijk % n[0])*5 + o[0]; if(i>=d[0]) return;
    ijk =  ijk / n[0];
    j   = (ijk % n[1])*5 + o[1]; if(j>=d[1]) return;
    k   = (ijk / n[1])*5 + o[2]; if(k>=d[2]) return;

    op1_dev(i, j, k, u, d, g, a, b, s);
}

__global__ void hu(float *r, const USIZE_t *d, const float *u, const float *a, const float *b, const float *s)
{
    USIZE_t ijk, tmp, i, j, k;
    ijk = threadIdx.x + blockDim.x*blockIdx.x;
    i   = ijk % d[0];
    tmp = ijk / d[0];
    j   = tmp % d[1];
    k   = tmp / d[1]; if(k>=d[2]) return;

    r[ijk] = hu_dev(i, j, k, d, u, a, b, s);
}

