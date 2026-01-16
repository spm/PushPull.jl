#include "cuheader.h"
/*
    % Gradients and Hessian from:
    syms p1 p2 real
    E = str2sym('(f1(x+u(x,p1,p2))*(1+diff(u(x,p1,p2),x)) - f2(x-u(x,p1,p2))*(1-diff(u(x,p1,p2),x)))^2/2')
    g = diff(E,p1)
    h = diff(E,p1,p2)

    This gives:
    g =  (f_1(x + u)*(D u + 1) + f_2(x - u)*(D u - 1))
       *((f_1(x + u) + f_2(x - u))*diff(u, p1, x) + (D(f_1)(x + u)*(D u + 1) - D(f_2)(x - u)*(D u - 1))*diff(u, p1))

    H = ((f_1(x + u) + f_2(x - u))*diff(u, p1, x) + (D(f_1)(x + u)*(D u + 1) - D(f_2)(x - u)*(D u - 1))*diff(u, p1))
       *((f_1(x + u) + f_2(x - u))*diff(u, p2, x) + (D(f_1)(x + u)*(D u + 1) - D(f_2)(x - u)*(D u - 1))*diff(u, p2))
      + (f_1(x + u)*(D u + 1) + f_2(x - u)*(D u - 1))
       *((f_2(x - u) + f_1(x + u) + D(f_1)(x + u)*(D u + 1) - D(f_2)(x - u)*(D u - 1))*diff(u, p1, p2)
        + (D(f_1)(x + u) - D(f_2)(x - u))*(diff(u, p2, x)*diff(u, p1) + diff(u, p1, x)*diff(u, p2))
        + (D(D(f_1))(x + u)*(D u + 1) + D(D(f_2))(x - u)*(D u - 1))*diff(u, p2)*diff(u, p1))

    For linear interpolation:
        D(D(f))(x + u)     = 0
        diff(u, p1, p2, x) = 0
    Other terms may become negative, so are omitted to increase robustness.

    This gives
    H = ((f_1(x + u) + f_2(x - u))*diff(u, p1, x) + (D(f_1)(x + u)*(D u + 1) - D(f_2)(x - u)*(D u - 1))*diff(u, p1))
       *((f_1(x + u) + f_2(x - u))*diff(u, p2, x) + (D(f_1)(x + u)*(D u + 1) - D(f_2)(x - u)*(D u - 1))*diff(u, p2))

   For linear interpolation, diff(u, p) is an identity matrix
   and diff(u, p, x) is a toeplitz matrix that computes gradients.


   %%%%%%%%%%%%%%%%%%%%%%%%%%%
   % The form of the likelihood Hessian
   % The general form is:
   M = 5;
   D = toeplitz([0 -1 zeros(1,M-3) 1]/2,[0 1 zeros(1,M-3) -1]/2);
   I = eye(M);
   c = sym('c',[M,1],'positive');
   b = sym('b',[M,1],'positive');
   a = sym('a',[M,1],'positive');
   S = b.*D + a.*I;
   g = S'*c; % Gradient
   H = S'*S; % Hessian
   h = H(3,:).'

   %%%%%%%%%%%%%%%%%%%%%%%%%%%
   % Updates are based on:
   % https://en.wikipedia.org/wiki/Jacobi_method
*/

#define TINY 1e-6

#define MOD(i,m) ((i)%(signed)(m)+(m))%(m)
#define BOUND(j,m) ((t_=MOD((signed)(j),(signed)(m)<<1))<(m) ? t_ : (((m)<<1)-1-t_)) /* reflect */
/* #define BOUND(j,m) MOD((signed)(j),(signed)(m)) */ /* circulant */

__device__ void blip_dev(USIZE_t i, USIZE_t j, USIZE_t k, const USIZE_t *d, float *u,
                        const float *g, const float *aa, const float *bb, const float *ab, const float *s)
{
    float t, v0 = s[0], v1 = s[1], v2 = s[2], w, w0, sv = v0+v1+v2, uii;
    SSIZE_t ii = i + d[0]*(j + d[1]*k), o1, o2, o4, o5;
    SSIZE_t t_;

    o1  = ii+(MOD((signed)j-2,d[1])-j)*d[0];
    o2  = ii+(MOD((signed)j-1,d[1])-j)*d[0];
    o4  = ii+(MOD((signed)j+1,d[1])-j)*d[0];
    o5  = ii+(MOD((signed)j+2,d[1])-j)*d[0];

    uii = u[ii]; /* Middle voxel */

    /* Likelihood part */
    t   = g[ii] - (((u[o1]-uii)* bb[o2]         + (u[o5]-uii)* bb[o4])/(-4)
                 + ((u[o2]-uii)*(ab[o2]-ab[ii]) + (u[o4]-uii)*(ab[ii]-ab[o4]))/2
                 + uii * (aa[ii]+(ab[o2]-ab[o4])/2));

    /* Begin to compute the denominator in the Jacobi update */
    w0  = aa[ii] + (bb[o2] + bb[o4])/4;

    /* Bending energy regularisation part */
    /* u[i,j-1,k] + u[i,j+1,k] */
    w0 -= 2*(w = -4*v1*sv);
    t  -= ((u[o2]-uii)+(u[o4]-uii))*w;

    /* u[i,j-2,k] + u[i,j+2,k] */
    w0 -= 2*(w = v1*v1);
    t  -= ((u[o1]-uii)+(u[o5]-uii))*w;

    o1  = BOUND(i-2,d[0])-i;
    o5  = BOUND(i+2,d[0])-i;

    /* u[i-2,j,k] + u[i-2,j,k] */
    w0 -= 2*(w = v0*v0);
    t  -= ((u[ii+o1]-uii) + (u[ii+o5]-uii))*w;

    o1  = (BOUND(k-2,d[2])-k)*d[0]*d[1];
    o5  = (BOUND(k+2,d[2])-k)*d[0]*d[1];

    /* u[i,j,k-2] + u[i,j,k+2] */
    w0 -= 2*(w = v2*v2);
    t  -= ((u[ii+o1]-uii) + (u[ii+o5]-uii))*w;

    o1  = BOUND(i-1,d[0])-i;
    o5  = BOUND(i+1,d[0])-i;

    /* u[i-1,j-1,k] + u[i+1,j-1,k] + u[i-1,j+1,k] + u[i+1,j+1,k] */
    w0 -= 4*(w = 2*v0*v1);
    t  -= ((u[o2+o1]-uii) + (u[o2+o5]-uii) + (u[o4+o1]-uii) + (u[o4+o5]-uii))*w;

    /* u[i-1,j,k] + u[i+1,j,k] */
    w0 -= 2*(w = -4*v0*sv);
    t  -= ((u[ii+o1]-uii) + (u[ii+o5]-uii))*w;

    o1  = (BOUND(k-1,d[2])-k)*d[0]*d[1];
    o5  = (BOUND(k+1,d[2])-k)*d[0]*d[1];

    /* u[i,j-1,k-1] + u[i,j-1,k+1] + u[i,j+1,k-1] + u[i,j+1,k+1] */
    w0 -= 4*(w = 2*v1*v2);
    t  -= ((u[o2+o1]-uii) + (u[o2+o5]-uii) + (u[o4+o1]-uii) + (u[o4+o5]-uii))*w;

    o2  = ii+o1;
    o4  = ii+o5;

    /* u[i,j,k-1] + u[i,j,k+1] */
    w0 -= 2*(w = -4*v2*sv);
    t  -= ((u[o2]-uii) + (u[o4]-uii))*w;

    o1 = BOUND(i-1,d[0])-i;
    o5 = BOUND(i+1,d[0])-i;

    /* u[i-1,j,k-1] + u[i+1,j,k-1] + u[i-1,j,k+1] + u[i+1,j,k+1] */
    w0 -= 4*(w = 2*v0*v2);
    t  -= ((u[o2+o1]-uii) + (u[o2+o5]-uii) + (u[o4+o1]-uii) + (u[o4+o5]-uii))*w;

    w      = TINY; /* *(sv*sv + TINY); */
    t     -= uii*w;
    u[ii] += t/(w0 + w);
}

__device__ void blip_nopad_dev(const USIZE_t *d, float *u,
                                const float *g, const float *aa, const float *bb, const float *ab, const float *s)
{
    float t, v0 = s[0], v1 = s[1], v2 = s[2], w, w0, sv = v0+v1+v2, u0;
    SSIZE_t o4, o5;

    o4  = d[0];
    o5  = 2*o4;
    u0  = u[0];
    t   = g[0] - (((u[-o5]-u0)* bb[-o4]        + (u[o5]-u0)* bb[o4])/(-4)
                + ((u[-o4]-u0)*(ab[-o4]-ab[0]) + (u[o4]-u0)*(ab[0]-ab[o4]))/2
                + u0 * (aa[0]+(ab[-o4]-ab[o4])/2));
    w0  = aa[0] + (bb[-o4] + bb[o4])/4;

    w0 -= 2*(w = -4*v1*sv);
    t  -= ((u[-o4]-u0)+(u[o4]-u0))*w;
    w0 -= 2*(w = v1*v1);
    t  -= ((u[-o5]-u0)+(u[o5]-u0))*w;

    o5  = d[0]*d[1];
    w0 -= 4*(w = 2*v0*v2);
    t  -= ((u[-o5-1] -u0) + (u[-o5+1] -u0) + (u[o5-1] -u0) + (u[o5+1] -u0))*w;
    w0 -= 4*(w = 2*v0*v1);
    t  -= ((u[-o4-1] -u0) + (u[-o4+1] -u0) + (u[o4-1] -u0) + (u[o4+1] -u0))*w;
    w0 -= 4*(w = 2*v1*v2);
    t  -= ((u[-o4-o5]-u0) + (u[-o4+o5]-u0) + (u[o4-o5]-u0) + (u[o4+o5]-u0))*w;

    w0 -= 2*(w = -4*v2*sv);
    t  -= ((u[-o5]-u0) + (u[o5]-u0))*w;
    w0 -= 2*(w = v0*v0);
    t  -= ((u[-2] -u0) + (u[2] -u0))*w;
    o5  = 2*o5;
    w0 -= 2*(w = v2*v2);
    t  -= ((u[-o5]-u0) + (u[o5]-u0))*w;
    w0 -= 2*(w = -4*v0*sv);
    t  -= ((u[-1] -u0) + (u[1] -u0))*w;

    w     = TINY; /* *(sv*sv + TINY); */
    t    -= u0*w;
    u[0] += t/(w0 + w);
}


__device__ float hu_dev(USIZE_t i, USIZE_t j, USIZE_t k, const USIZE_t *d, const float *u,
                        const float *aa, const float *bb, const float *ab, const float *s)
{
    float t, v0 = s[0], v1 = s[1], v2 = s[2], w, sv = v0+v1+v2, uii;
    SSIZE_t ii = i + d[0]*(j + d[1]*k), o1, o2, o4, o5;
    SSIZE_t t_;

    o1 = ii+(MOD((signed)j-2,(signed)d[1])-(signed)j)*(signed)d[0];
    o2 = ii+(MOD((signed)j-1,(signed)d[1])-(signed)j)*(signed)d[0];
    o4 = ii+(MOD((signed)j+1,(signed)d[1])-(signed)j)*(signed)d[0];
    o5 = ii+(MOD((signed)j+2,(signed)d[1])-(signed)j)*(signed)d[0];

    uii = u[ii];
    if (aa != (const float *)NULL)
        t   = (u[o1]-uii)*(-bb[o2]/4)       + (u[o5]-uii)*(-bb[o4]/4)
            + (u[o2]-uii)*(ab[o2]-ab[ii])/2 + (u[o4]-uii)*(ab[ii]-ab[o4])/2
            + uii * (aa[ii]+(ab[o2]-ab[o4])/2);
    else
        t = 0.0;

    /* u[i,j-1,k] + u[i,j+1,k] */
    w   = -4*v1*sv;
    t  += ((u[o2]-uii) + (u[o4]-uii))*w;

    /* u[i,j-2,k] + u[i,j+2,k] */
    w   = v1*v1;
    t  += ((u[o1]-uii) + (u[o5]-uii))*w;

    o1  = BOUND(i-2,d[0])-i;
    o5  = BOUND(i+2,d[0])-i;

    /* u[i-2,j,k] + u[i-2,j,k] */
    w   = v0*v0;
    t  += ((u[ii+o1]-uii) + (u[ii+o5]-uii))*w;

    o1  = (BOUND(k-2,d[2])-k)*d[0]*d[1];
    o5  = (BOUND(k+2,d[2])-k)*d[0]*d[1];

    /* u[i,j,k-2] + u[i,j,k+2] */
    w   = v2*v2;
    t  += ((u[ii+o1]-uii) + (u[ii+o5]-uii))*w;

    o1  = BOUND(i-1,d[0])-i;
    o5  = BOUND(i+1,d[0])-i;

    /* u[i-1,j-1,k] + u[i+1,j-1,k] + u[i-1,j+1,k] + u[i+1,j+1,k] */
    w   = 2*v0*v1;
    t  += ((u[o2+o1]-uii) + (u[o2+o5]-uii) + (u[o4+o1]-uii) + (u[o4+o5]-uii))*w;

    /* u[i-1,j,k] + u[i+1,j,k] */
    w   = -4*v0*sv;
    t  += ((u[ii+o1]-uii) + (u[ii+o5]-uii))*w;

    o1  = (BOUND(k-1,d[2])-k)*d[0]*d[1];
    o5  = (BOUND(k+1,d[2])-k)*d[0]*d[1];

    /* u[i,j-1,k-1] + u[i,j-1,k+1] + u[i,j+1,k-1] + u[i,j+1,k+1] */
    w   = 2*v1*v2;
    t  += ((u[o2+o1]-uii) + (u[o2+o5]-uii) + (u[o4+o1]-uii) + (u[o4+o5]-uii))*w;

    o2  = ii+o1;
    o4  = ii+o5;

    /* u[i,j,k-1] + u[i,j,k+1] */
    w   = -4*v2*sv;
    t  += ((u[o2]-uii) + (u[o4]-uii))*w;

    o1  = BOUND(i-1,d[0])-i;
    o5  = BOUND(i+1,d[0])-i;

    /* u[i-1,j,k-1] + u[i+1,j,k-1] + u[i-1,j,k+1] + u[i+1,j,k+1] */
    w   = 2*v0*v2;
    t  += ((u[o2+o1]-uii) + (u[o2+o5]-uii) + (u[o4+o1]-uii) + (u[o4+o5]-uii))*w;

    w   = TINY; /* *(sv*sv + TINY); */
    t  += uii*w;

    return(t);
}

