#include "cuheader.h"
#include "sparse_operator_dev.cu"

void vel2mom_midd(USIZE_t i_start, USIZE_t i_stop, USIZE_t j_start, USIZE_t j_stop, USIZE_t k_start, USIZE_t k_stop,
                  float *u, const USIZE_t *d, const float* v,
                  const int *offset, const int *length, const float *values, const int *indices)
{
    USIZE_t k;
    for(k=k_start; k<k_stop; k++)
    {
        USIZE_t j, kd = d[1]*k;
        for(j=j_start; j<j_stop; j++)
        {
            USIZE_t i, jkd = d[0]*(j + kd);
            for(i=i_start; i<i_stop; i++)
            {
                USIZE_t ijk = jkd + i;
                vel2mom1(u+ijk, d, v+ijk,  offset, length, values, indices);
            }
        }
    }
}

void vel2mom_edge(USIZE_t i_start, USIZE_t i_stop, USIZE_t j_start, USIZE_t j_stop, USIZE_t k_start, USIZE_t k_stop,
                   float *u, const USIZE_t *d, const float* v,
                   const int *offset, const int *length, const float *values, const int *patch_indices,
                   const USIZE_t *dp, const int *bnd)
{
    USIZE_t i, j, k;
    for(k=k_start; k<k_stop; k++)
        for(j=j_start; j<j_stop; j++)
            for(i=i_start; i<i_stop; i++)
                vel2mom_padded1(i,j,k, u, d, v,  offset, length, values, patch_indices, dp, bnd);
}

#define MIN(a,b) ((signed)(a)<(signed)(b) ? (a) : (b))
#define MAX(a,b) ((signed)(a)>(signed)(b) ? (a) : (b))

void vel2mom(float *u, const float* v, const USIZE_t *d, const USIZE_t *dp,
             const int *offset, const int *length,
             const float *values, const int *indices, const int *patch_indices, const int *bnd)
{
    USIZE_t rs[3], re[3], i;
    for(i=0; i<3; i++)
    {
        rs[i] = MIN((dp[i]-1)/2,d[i]);
        re[i] = MAX(rs[i],d[i]-(dp[i]-1)/2);
    }
    vel2mom_edge(   0 , d[0],    0 , d[1],    0 , rs[2], u, d, v, offset, length, values, patch_indices, dp, bnd);
    vel2mom_edge(   0 , d[0],    0 ,rs[1], rs[2], re[2], u, d, v, offset, length, values, patch_indices, dp, bnd);
    vel2mom_edge(   0 ,rs[0], rs[1],re[1], rs[2], re[2], u, d, v, offset, length, values, patch_indices, dp, bnd);
    vel2mom_midd(rs[0],re[0], rs[1],re[1], rs[2], re[2], u, d, v, offset, length, values, indices);
    vel2mom_edge(re[0], d[0], rs[1],re[1], rs[2], re[2], u, d, v, offset, length, values, patch_indices, dp, bnd);
    vel2mom_edge(   0 , d[0], re[1], d[1], rs[2], re[2], u, d, v, offset, length, values, patch_indices, dp, bnd);
    vel2mom_edge(   0 , d[0],    0 , d[1], re[2],  d[2], u, d, v, offset, length, values, patch_indices, dp, bnd);
}


