#define CUDA
#include "cuheader.h"
#include "blip_dev.cu"

__global__ void blip(float *u, const USIZE_t *d, const float *g,
                     const float *aa, const float *bb, const float *ab,
                     const float *s, const USIZE_t *o, const USIZE_t *n)
{
    USIZE_t ijk, i, j, k;
    ijk = threadIdx.x + blockDim.x*blockIdx.x;
    i   = (ijk % n[0])*3 + o[0]; if(i>=d[0]) return;
    ijk =  ijk / n[0];
    j   = (ijk % n[1])*3 + o[1]; if(j>=d[1]) return;
    k   = (ijk / n[1])*3 + o[2]; if(k>=d[2]) return;

    blip_dev(i, j, k, d, u, g, aa, bb, ab, s);
}

__global__ void hu(float *r, const USIZE_t *d, const float *u,
                   const float *aa, const float *bb, const float *ab, const float *s)
{
    USIZE_t ijk, tmp, i, j, k;
    ijk = threadIdx.x + blockDim.x*blockIdx.x;
    i   = ijk % d[0];
    tmp = ijk / d[0];
    j   = tmp % d[1];
    k   = tmp / d[1]; if(k>=d[2]) return;

    r[ijk] = hu_dev(i, j, k, d, u, aa, bb, ab, s);
}

