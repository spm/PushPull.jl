#include "cuheader.h"
#include "sparse_operator_dev.cu"

void vel2mom_midd(USIZE_t i_start, USIZE_t i_stop, USIZE_t j_start, USIZE_t j_stop, USIZE_t k_start, USIZE_t k_stop,
                  float *u, const USIZE_t *d, const float* v,
                  const int *offset, const int *length, const float *values, const int *indices)
{
    SSIZE_t k;
    for(k=k_start; k<k_stop; k++)
    {
        SSIZE_t j, kd = d[1]*k;
        for(j=j_start; j<j_stop; j++)
        {
            SSIZE_t i, jkd = d[0]*(j + kd);
            for(i=i_start; i<i_stop; i++)
            {
                USIZE_t ijk = jkd + i;
                vel2mom1(u+ijk, d, v+ijk,  offset, length, values, indices);
            }
        }
    }
}

/*
vel2mom_padded1(USIZE_t i, USIZE_t j, USIZE_t k, float *u, const USIZE_t *d, const float *v,
                const int *offset, const int *length, const float *values, const int *patch_indices,
                const USIZE_t *dp, const int *bnd)
*/
void vel2mom_edge(USIZE_t i_start, USIZE_t i_stop, USIZE_t j_start, USIZE_t j_stop, USIZE_t k_start, USIZE_t k_stop,
                   float *u, const USIZE_t *d, const float* v,
                   const int *offset, const int *length, const float *values, const int *patch_indices,
                   const USIZE_t *dp, const int *bnd)
{
    SSIZE_t i, j, k;
    for(k=k_start; k<k_stop; k++)
        for(j=j_start; j<j_stop; j++)
            for(i=i_start; i<i_stop; i++)
                vel2mom_padded1(i,j,k, u, d, v,  offset, length, values, patch_indices, dp, bnd);
}


#define MIN(a,b) ((signed)(a)<(signed)(b) ? (a) : (b))
#define MAX(a,b) ((signed)(a)>(signed)(b) ? (a) : (b))

void vel2mom(float *u, const float* v, const USIZE_t *d,
             const int *offset, const int *length, const float *values,
             const int *indices, const int *patch_indices, const USIZE_t *dp, const int *bnd)
{
    SSIZE_t rs[3], re[3], i;
    for(i=0; i<3; i++)
    {
        rs[i] = MIN((dp[i]+1)/2,d[i]);
        re[i] = MAX(rs[i],(signed)d[i]-(dp[i]+1)/2);
    }
 /* vel2mom_edge(   0 , d[0],    0 , d[1],    0 ,  d[2], u, d, v, offset, length, values, patch_indices, dp, bnd); */
    vel2mom_edge(   0 , d[0],    0 , d[1],    0 , rs[2], u, d, v, offset, length, values, patch_indices, dp, bnd);
    vel2mom_edge(   0 , d[0],    0 ,rs[1], rs[2], re[2], u, d, v, offset, length, values, patch_indices, dp, bnd);
    vel2mom_edge(   0 ,rs[0], rs[1],re[1], rs[2], re[2], u, d, v, offset, length, values, patch_indices, dp, bnd);
    vel2mom_midd(rs[0],re[0], rs[1],re[1], rs[2], re[2], u, d, v, offset, length, values, indices);
    vel2mom_edge(re[0], d[0], rs[1],re[1], rs[2], re[2], u, d, v, offset, length, values, patch_indices, dp, bnd);
    vel2mom_edge(   0 , d[0], re[1], d[1], rs[2], re[2], u, d, v, offset, length, values, patch_indices, dp, bnd);
    vel2mom_edge(   0 , d[0],    0 , d[1], re[2],  d[2], u, d, v, offset, length, values, patch_indices, dp, bnd);
}


void relax_edge(const SSIZE_t i_start, const SSIZE_t i_stop,
                const SSIZE_t j_start, const SSIZE_t j_stop,
                const SSIZE_t k_start, const SSIZE_t k_stop,
                float *v, const USIZE_t *d, const float *g, const float *h,
                const int *offset, const int *length, const float *values, const int *patch_indices,
                const USIZE_t *dp, const int *bnd)
{
    SSIZE_t i0, j0, k0;
    for(k0=k_start; k0<MIN(k_stop, k_start+dp[2]); k0++)
        for(j0=j_start; j0<MIN(j_stop, j_start+dp[1]); j0++)
            for(i0=i_start; i0<MIN(i_stop, i_start+dp[0]); i0++)
            {
                SSIZE_t i, j, k;
                for(k=k0; k<k_stop; k+=dp[2])
                    for(j=j0; j<j_stop; j+=dp[1])
                        for(i=i0; i<i_stop; i+=dp[0])
                            relax_padded1(i,j,k, v, d, g, h, offset, length, values, patch_indices, dp, bnd);
            }
}


void relax_midd(const SSIZE_t i_start, const SSIZE_t i_stop,
                const SSIZE_t j_start, const SSIZE_t j_stop,
                const SSIZE_t k_start, const SSIZE_t k_stop,
                float *v, const USIZE_t *d, const float *g, const float *h,
                const int *offset, const int *length, const float *values, const int *indices,
                const USIZE_t *dp)
{
    SSIZE_t i0, j0, k0;
    for(k0=k_start; k0<MIN(k_stop, k_start+dp[2]); k0++)
        for(j0=j_start; j0<MIN(j_stop, j_start+dp[1]); j0++)
            for(i0=i_start; i0<MIN(i_stop, i_start+dp[0]); i0++)
            {
                SSIZE_t i, j, k;
                for(k=k0; k<k_stop; k+=dp[2])
                {
                    SSIZE_t ok = k*d[1];
                    for(j=j0; j<j_stop; j+=dp[1])
                    {
                        SSIZE_t oj = d[0]*(j + ok);
                        for(i=i0; i<i_stop; i+=dp[0])
                        {
                            SSIZE_t oi = oj + i;
                            relax1(v+oi, d, g+oi, h+oi, offset, length, values, indices);
                        }
                    }
                }
            }
}

void relax(float *v, const USIZE_t *d, const float *g, const float *h,
           const int *offset, const int *length, const float *values, const int *indices, const int *patch_indices,
           const USIZE_t *dp, const int *bnd)
{
    USIZE_t i0, i1, j0, j1, k0, k1;
    i0 = MIN(3,  d[0]);
    j0 = MIN(3,  d[1]);
    k0 = MIN(3,  d[2]);
    i1 = MAX(i0, d[0]-(dp[0]+1)/2);
    j1 = MAX(j0, d[1]-(dp[1]+1)/2);
    k1 = MAX(k0, d[2]-(dp[2]+1)/2);

    /* Note that results are not identical to those from the CUDA
     * implementation because of the ordering of the Gauss–Seidel
     * updates. Maybe (just maybe) fix this later. */
    relax_edge( 0,d[0],  0,d[1],  0,  k0, v, d, g, h, offset, length, values, patch_indices, dp, bnd);
    relax_edge( 0,d[0],  0,  j0, k0,  k1, v, d, g, h, offset, length, values, patch_indices, dp, bnd);
    relax_edge( 0,  i0, j0,  j1, k0,  k1, v, d, g, h, offset, length, values, patch_indices, dp, bnd);
    relax_midd(i0,  i1, j0,  j1, k0,  k1, v, d, g, h, offset, length, values, indices, dp);
    relax_edge(i1,d[0], j0,  j1, k0,  k1, v, d, g, h, offset, length, values, patch_indices, dp, bnd);
    relax_edge( 0,d[0], j1,d[1], k0,  k1, v, d, g, h, offset, length, values, patch_indices, dp, bnd);
    relax_edge( 0,d[0],  0,d[1], k1,d[2], v, d, g, h, offset, length, values, patch_indices, dp, bnd);
}

