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
    float f0 = f[indices[0]], s = 0.0;
    for(j=1; j<len; j++)
        s += values[j]*(f[indices[j]]-f0);
    s += values[-1]*f0;
    return(s);
}

__device__ void Hv(float *b, USIZE_t d3, USIZE_t d4, USIZE_t nd, const float *h, const float *v)
{
    if(d4)
    {
        int ii;
        for(ii=0; ii<d3; ii++)
            b[ii] -= h[nd*ii]*v[nd*ii];

        if(d4==2)
        {
            int j;
            for(j=0, ii=d3; j<d3; j++)
            {
                int i;
                for(i=j+1; i<d3; i++, ii++)
                {
                    b[j] -= h[ii*nd]*v[nd*i];
                    b[i] -= h[ii*nd]*v[nd*j];
                }
            }
        }
    }
}

__device__ void make_A(float *A, USIZE_t d3, USIZE_t d4, USIZE_t nd, const float *h, const float *values, const int *offset)
{
    int j;
    for(j=0; j<d3; j++)
    {
        int i, j3 = j*d3, ij = j3+j;
        A[ij] = values[offset[ij]]*1.000001f;
        for(i=j+1; i<d3; i++)
        {
            ij    = i+j3;
            A[ij] = A[i*d3+j] = values[offset[ij]];
        }
    }

    if(d4)
    {
        int ii;
        for(ii=0; ii<d3; ii++)
            A[ii+d3*ii] += h[nd*ii]*1.000001f;

        if(d4==2)
        {
            for(j=0, ii=d3; j<d3; j++)
            {
                int i;
                for(i=j+1; i<d3; i++, ii++)
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
    USIZE_t m = i+d[0]*(j+d[1]*k), nd = d[0]*d[1]*d[2], d3 = d[3];
    float patch[BUFLEN];
    float *A = patch, *p = patch + MAXN*MAXN, *x = patch+MAXN*(MAXN+1); /* re-use memory */
    float b[MAXN];
    SSIZE_t off[3];

    off[0]  = (SSIZE_t)i-(SSIZE_t)dp[0]/2;
    off[1]  = (SSIZE_t)j-(SSIZE_t)dp[1]/2;
    off[2]  = (SSIZE_t)k-(SSIZE_t)dp[2]/2;

    g   += m;
    h   += m;

    /* Original i, j & k no-longer needed, so variables are re-used */
    for(i=0; i<d3; i++) b[i] = g[nd*i];
    for(j=0; j<d3; j++)
    {
        int j3 = d3*j;
        get_patch(dp, patch, bnd+3*j, off, d, v+j*nd);
        for(i=0; i<d3; i++)
        {
            int ij = i+j3;
            b[i]  -= sp_conv(length[ij], patch_indices+offset[ij], values+offset[ij], patch);
        }
    }

    v += m; /* shift pointer */
    Hv(b, d3, d[4], nd, h, v);

    /* Construct "diagonal" of L+H, and re-use the "patch" memory */
    make_A(A, d3, d[4], nd, h, values, offset);

    /* Compute x = A\b via Cholesky decomposition */
    choldcf(d3, A, p);
    chollsf(d3, A, p, b, x);

    for(i=0; i<d3; i++, v+=nd) *v += x[i];
}


__device__ void relax1(float *v, const USIZE_t *d, const float *g, const float *h,
                      const int *offset, const int *length, const float *values, const int *indices)
{
    USIZE_t nd = d[0]*d[1]*d[2], d3 = d[3], i, j;
    float A[MAXN*MAXN], p[MAXN], x[MAXN], b[MAXN];

    for(i=0; i<d3; i++) b[i] = g[nd*i];
    for(j=0; j<d3; j++)
    {
        int j3 = j*d3;
        for(i=0; i<d3; i++)
        {
            int ij = i+j3;
            b[i]  -= sp_conv(length[ij], indices+offset[ij], values+offset[ij], v);
        }
    }

    Hv(b, d3, d[4], nd, h, v);

    /* Construct "diagonal" of L+H */
    make_A(A, d3, d[4], nd, h, values, offset);

    /* Compute x = A\b via Cholesky decomposition */
    choldcf(d3, A, p);
    chollsf(d3, A, p, b, x);

    for(i=0; i<d3; i++, v+=nd) *v += x[i];
}


__device__ void vel2mom1(float *u, const USIZE_t *d, const float *v,
                         const int *offset, const int *length, const float *values, const int *indices)
{
    int d3 = d[3], i;

    for(i=0; i<d3; i++)
    {
        int j;
        float t = 0.0;
        for(j=0; j<d3; j++)
        {
            int ij = i+d3*j;
            t     += sp_conv(length[ij], indices+offset[ij], values+offset[ij], v);
        }
        u[indices[offset[i*d3]]] = t;
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
        get_patch(dp, patch, bnd+3*j, off, d, v+j*nd);
        for(i=0; i<d3; i++)
        {
            int ij = i+j*d3;
            int o  = offset[ij];
            u[m + i*nd] += sp_conv(length[ij], patch_indices+o, values+o, patch);
        }
    }
}

