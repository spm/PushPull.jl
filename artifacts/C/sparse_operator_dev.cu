#include "cuheader.h"
#include "patch.cu"
#include "chol.cu"
/* Flexibility comes at the expense of speed because L is often mostly zeros.
   Dealing with padding is also likely to slow things down a lot, especially
   as many of the loaded voxels will not be used. */

#define MAXN 8
#define MAXD 5
#define BUFLEN 125 /* Maximum of MAXD*MAXD*MAXD and MAXN*(MAXN+2) */

__device__ float sp_conv(USIZE_t len, const int indices[], const float values[], const float *f)
{
    int j;
    float f0 = f[indices[0]], s = 0.0;
    for(j=1; j<len; j++)
        s += values[j]*(f[indices[j]]-f0);
    s += (values[0]+values[-1])*f0;
    return(s);
}

__device__ float sp_odconv(USIZE_t len, const int indices[], const float values[], const float *f)
{
    int j;
    float f0 = f[indices[0]], s = values[-1]*f0;
    for(j=1; j<len; j++)
        s += values[j]*(f[indices[j]]-f0);
    return(s);
}

__device__ void make_A(float *A, USIZE_t d3, USIZE_t d4, USIZE_t nd, const float *h, const float *values, const int *offset)
{
    USIZE_t i;

    for(i=0; i<d3; i++)
    {
        USIZE_t j;
        A[i+d3*i] = values[offset[i*d3+i]]*1.000001f;
        for(j=i+1; j<d3; j++)
            A[i + j*d3] = A[j + i*d3] = values[offset[j*d3+i]];
    }

    if(d4)
    {
        USIZE_t ii;
        for(i=0; i<d3; i++)
            A[i+d3*i] += h[nd*i]*1.000001f;

        if(d4==2)
        {
            for(i=0, ii=d3; i<d3; i++)
            {
                USIZE_t j;
                for(j=i+1; j<d3; j++, ii++)
                    A[i + j*d3] = A[j + i*d3] += h[ii*nd];
            }
        }
    }
}


__device__ void relax_padded1(USIZE_t i, USIZE_t j, USIZE_t k,
                             float *v, const USIZE_t *d, const float *g, const float *h,
                             const int *offset, const int *length, const float *values, const int *patch_indices,
                             const USIZE_t *dp, const int *bnd)
{
    USIZE_t m, nd, d3 = d[3];
    float patch[BUFLEN];
    float *A = patch, *p = patch + MAXN*MAXN, *x = patch+MAXN*(MAXN+1); /* re-use memory */
    float b[MAXN];
    SSIZE_t off[3];

    off[0]  = (SSIZE_t)i-(SSIZE_t)dp[0]/2;
    off[1]  = (SSIZE_t)j-(SSIZE_t)dp[1]/2;
    off[2]  = (SSIZE_t)k-(SSIZE_t)dp[2]/2;

    m    = i+d[0]*(j+d[1]*k);
    g   += m;
    h   += m;

    /* Original i, j & k no-longer needed, so variables are re-used */

    nd   = d[0]*d[1]*d[2];

    for(j=0; j<d3; j++) b[j] = g[nd*j];
    for(i=0; i<d3; i++)
    {
        get_patch(dp, patch, bnd+i*3, off, d, v+i*nd);
        for(j=0; j<d3; j++)
            b[j] -= sp_odconv(length[j*d3+i], patch_indices+offset[j*d3+i], values+offset[j*d3+i], patch);
    }

    /* Construct "diagonal" of L+H, and re-use the "patch" memory */
    make_A(A, d3, d[4], nd, h, values, offset);

    /* Compute x = A\b via Cholesky decomposition */
    choldcf(d3, A, p);
    chollsf(d3, A, p, b, x);

    v += m; /* shift pointer */
    for(i=0; i<d3; i++, v+=nd) *v = x[i];
}


__device__ void relax1(float *v, const USIZE_t *d, const float *g, const float *h,
                      const int *offset, const int *length, const float *values, const int *indices)
{
    USIZE_t nd = d[0]*d[1]*d[2], d3 = d[3], i, j;
    float A[MAXN*MAXN], p[MAXN], x[MAXN], b[MAXN];

    for(j=0; j<d3; j++) b[j] = g[nd*j];
    for(i=0; i<d3; i++)
    {
        for(j=0; j<d3; j++)
            b[j] -= sp_odconv(length[j*d3+i], indices+offset[j*d3+i], values+offset[j*d3+i], v);
    }

    /* Construct "diagonal" of L+H, and re-use the "patch" memory */
    make_A(A, d3, d[4], nd, h, values, offset);

    /* Compute x = A\b via Cholesky decomposition */
    choldcf(d3, A, p);
    chollsf(d3, A, p, b, x);

    for(i=0; i<d3; i++, v+=nd) *v = x[i];
}


__device__ void vel2mom1(float *u, const USIZE_t *d, const float *v,
                         const int *offset, const int *length, const float *values, const int *indices)
{
    USIZE_t d3 = d[3], j;

    for(j=0; j<d3; j++)
    {
        USIZE_t i;
        int  *o = (int *)offset + j*d3;
        float t = 0.0;
        for(i=0; i<d3; i++)
            t += sp_conv(length[j*d3+i], indices+o[i], values+o[i], v);
        u[indices[offset[j]]] = t;
    }
}

__device__ void vel2mom_padded1(USIZE_t i, USIZE_t j, USIZE_t k, float *u, const USIZE_t *d, const float *v,
                                const int *offset, const int *length, const float *values, const int *patch_indices,
                                const USIZE_t *dp, const int *bnd)
{
    USIZE_t d3 = d[3], nd = d[0]*d[1]*d[2], m = i+d[0]*(j+d[1]*k);
    float patch[BUFLEN];
    SSIZE_t off[3];

    off[0]  = (SSIZE_t)i-(SSIZE_t)dp[0]/2;
    off[1]  = (SSIZE_t)j-(SSIZE_t)dp[1]/2;
    off[2]  = (SSIZE_t)k-(SSIZE_t)dp[2]/2;

    for(j=0; j<d3; j++)
    {
        int  *o = (int *)offset + j*d3;
        float t = 0.0;
        for(i=0; i<d3; i++)
        {
            get_patch(dp, patch, bnd+i*3, off, d, v+i*nd);
            t += sp_conv(length[j*d3+i], patch_indices+o[i], values+o[i], patch);
        }
        u[m + j*nd] = t;
    }
}

