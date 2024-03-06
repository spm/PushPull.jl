#define CUDA
#include "cuheader.h"
#include "sparse_operator_dev.cu"
/*
    MAXD - maximum filter size
    MAXN - maximum number of gradient fields
*/

/*
    Use constant memory for faster access times
    Limited by available constant memory on device:
        CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY
*/

#define MAX_ELEM 256

__constant__ float   values[MAX_ELEM];         /* Values in sparse matrix*/
__constant__ int     indices[MAX_ELEM];        /* Indices into images */
__constant__ int     patch_indices[MAX_ELEM];  /* Indices into patchs */
__constant__ int     offset[MAXN*MAXN];        /* Offsets into values/indices */
__constant__ int     length[MAXN*MAXN];        /* Number of non-zero off diagonal elements */
__constant__ int     bnd[3*MAXN];         /* Boundary conditions */
__constant__ USIZE_t  d[5];               /* image data dimensions */
__constant__ USIZE_t dp[3];               /* filter dimensions */
__constant__ USIZE_t  o[3];               /* offsets into volume */
__constant__ USIZE_t  n[3];               /* number of elements */


__global__ void relax_padded_element(float *v, const float *g, const float *h)
{
    USIZE_t ijk, i, j, k;
    ijk = threadIdx.x + blockDim.x*blockIdx.x;
    i   = (ijk % n[0])*dp[0] + o[0];
    ijk =  ijk / n[0];
    j   = (ijk % n[1])*dp[1] + o[1];
    k   =  ijk / n[1];
    if(k>=n[2]) return;
    k   = k*dp[2] + o[2];
    relax_padded1(i,j,k, v, d, g, h, offset, length, values, patch_indices, dp, bnd);
}

__global__ void relax_element(float *v, const float *g, const float *h)
{
    USIZE_t ijk, i, j, k;
    ijk = threadIdx.x + blockDim.x*blockIdx.x;
    i   = (ijk % n[0])*dp[0] + o[0];
    ijk =  ijk / n[0];
    j   = (ijk % n[1])*dp[1] + o[1];
    k   =  ijk / n[1];
    if(k>=n[2]) return;
    k   = k*dp[2] + o[2];
    ijk = i + d[0]*(j + d[1]*k);
    relax1(v+ijk, d, g+ijk, h+ijk, offset, length, values, indices);
}

__global__ void vel2mom_element(float *u, const float *v)
{
    USIZE_t ijk, i, j, k;
    ijk = threadIdx.x + blockDim.x*blockIdx.x;
    i   = (ijk % n[0]) + o[0];
    ijk =  ijk / n[0];
    j   = (ijk % n[1]) + o[1];
    k   =  ijk / n[1];
    if(k>=n[2]) return;
    k  += o[2];
    ijk = i + d[0]*(j + d[1]*k);
    vel2mom1(u+ijk, d, v+ijk,  offset, length, values, indices);
}

__global__ void vel2mom_padded_element(float *u, const float *v)
{
    USIZE_t ijk, i, j, k;
    ijk = threadIdx.x + blockDim.x*blockIdx.x;
    i   = (ijk % n[0]) + o[0];
    ijk =  ijk / n[0];
    j   = (ijk % n[1]) + o[1];
    k   =  ijk / n[1];
    if(k>=n[2]) return;
    k  += o[2];
    vel2mom_padded1(i,j,k, u, d, v,  offset, length, values, patch_indices, dp, bnd);
}

