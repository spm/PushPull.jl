#include "cuheader.h"
#include "blip_dev.cu"

#define min(a,b) ((a)<(b) ? (a) : (b))
#define max(a,b) ((a)>(b) ? (a) : (b))


void blip_pad(const USIZE_t i_start, const USIZE_t j_start, const USIZE_t k_start,
              const USIZE_t i_stop,  const USIZE_t j_stop,  const USIZE_t k_stop,
              float *u, const USIZE_t *d, const float *g,
              const float *aa, const float *bb, const float *ab,
              const float *s)
{
    USIZE_t i0, j0, k0;
    for(k0=k_start; k0<5; k0++)
        for(j0=j_start; j0<5; j0++)
            for(i0=i_start; i0<5; i0++)
            {
                USIZE_t i, j, k;
                for(k=k0; k<k_stop; k+=5)
                    for(j=j0; j<j_stop; j+=5)
                        for(i=i0; i<i_stop; i+=5)
                            blip_dev(i, j, k, d, u, g, aa, bb, ab, s);
            }
}


void blip_nopad(const USIZE_t i_start, const USIZE_t j_start, const USIZE_t k_start,
                const USIZE_t i_stop,  const USIZE_t j_stop,  const USIZE_t k_stop,
                float *u, const USIZE_t *d, const float *g,
                const float *aa, const float *bb, const float *ab,
                const float *s)
{
    USIZE_t i0, j0, k0;
    for(k0=k_start; k0<5; k0++)
        for(j0=j_start; j0<5; j0++)
            for(i0=i_start; i0<5; i0++)
            {
                USIZE_t i, j, k;
                for(k=k0; k<k_stop; k+=5)
                {
                    USIZE_t ok = k*d[1];
                    for(j=j0; j<j_stop; j+=5)
                    {
                        USIZE_t oj = d[0]*(j + ok); 
                        for(i=i0; i<i_stop; i+=5)
                        {
                            USIZE_t oi = oj + i;
                            blip_nopad_dev(d, u+oi, g+oi, aa+oi, bb+oi, ab+oi, s);
                        }
                    }
                }
            }
}


void blip(float *u, const USIZE_t *d, const float *g,
          const float *aa, const float *bb, const float *ab,
          const float *s)
{
    USIZE_t i0, i1, j0, j1, k0, k1;
    i0 = min(3,  d[0]);
    j0 = min(3,  d[1]);
    k0 = min(3,  d[2]);
    i1 = max(i0, d[0]-3);
    j1 = max(j0, d[1]-3);
    k1 = max(k0, d[2]-3);

    blip_pad(0,   0,  0, d[0], d[1], d[2], u, d, g, aa, bb, ab, s);
    /*
    blip_pad(0 ,  0,  0, d[0], d[1],   k0, u, d, g, aa, bb, ab, s);
    blip_pad(0,   0, k0, d[0],   j0,   k1, u, d, g, aa, bb, ab, s);
    blip_pad(0,  j0, k0,   i0,   j1,   k1, u, d, g, aa, bb, ab, s);
    blip_nopad(i0, j0, k0, i1,   j1,   k1, u, d, g, aa, bb, ab, s);
    blip_pad(i1, j0, k0, d[0],   j1,   k1, u, d, g, aa, bb, ab, s);
    blip_pad(0,  j1, k0, d[0], d[1],   k1, u, d, g, aa, bb, ab, s);
    blip_pad(0,   0, k1, d[0], d[1], d[2], u, d, g, aa, bb, ab, s);
    */
}


void hu(float *r, const USIZE_t *d, const float *u,
                   const float *aa, const float *bb, const float *ab, const float *s)
{
    USIZE_t i, j, k;
    for(k=0; k<d[2]; k++)
    {
        USIZE_t ok = k*d[1];
        for(j=0; j<d[1]; j++)
        {
            USIZE_t oj = d[0]*(j + ok);
            for(i=0; i<d[0]; i++)
            {
                r[oj+i]      = hu_dev(i, j, k, d, u, aa, bb, ab, s);
            }
        }
    }
}

