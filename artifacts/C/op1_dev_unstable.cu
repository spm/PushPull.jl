#include "cuheader.h"
/*
    % Gradients and Hessian from:
    syms p1 p2 real
    E = str2sym('(f1(x+u(x,p1,p2))*(1+diff(u(x,p1,p2),x)) - f2(x-u(x,p1,p2))*(1-diff(u(x,p1,p2),x)))^2/2')
    g = diff(E,p1)
    h = diff(E,p1,p2)

    This gives:
    g = (f_1(x + u)*(D u + 1) + f_2(x - u)*(D u - 1))
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


   % The general form is:
   M = 5;
   D = toeplitz([0 1 zeros(1,M-3) -1]/2,[0 -1 zeros(1,M-3) 1]/2);
   I = eye(M);
   b = sym('b',[M,1],'positive');
   a = sym('a',[M,1],'positive');
   A = b.*D + a.*I;
   H = A'*A; % Hessian

   % For Gauss-Siedel updates:
   g = sym('g',[M,1],'positive');
   u = sym('u',[M,1],'positive');

   u_new = diag(diag(H))\(g - (H-diag(diag(H)))*u);
   u_new(3)


*/

#define MOD(i,m) ((i)%(m)+(m))%(m)
#define BOUND(j,m) ((t_=MOD((signed)(j),(signed)(m)<<1))<(m) ? t_ : (((m)<<1)-1-t_))

__device__ void op1_dev(USIZE_t i, USIZE_t j, USIZE_t k, float *u, const USIZE_t *d,
                        const float *g, const float *a, const float *b, const float *s)
{
    float t, v0 = s[0]*s[0], v1 = s[1]*s[1], v2 = s[2]*s[2], w, w0, sv = v0+v1+v2;
    SSIZE_t ii = i + d[0]*(j + d[1]*k), o1, o2, o4, o5;
    SSIZE_t t_;

    o1  = ii+(BOUND(j-2,d[1])-j)*d[0];
    o2  = ii+(BOUND(j-1,d[1])-j)*d[0];
    o4  = ii+(BOUND(j+1,d[1])-j)*d[0];
    o5  = ii+(BOUND(j+2,d[1])-j)*d[0];

    t   = g[ii] + (u[o2]*(b[o2]*a[o2] - b[ii]*a[ii]) - u[o4]*(b[o4]*a[o4] - b[ii]*a[ii])
                + (u[o1]*(b[o2]*b[o2])               + u[o5]*(b[o4]*b[o4]))/2)/2;
    w0  = a[ii]*a[ii] + (b[o2]*b[o2] + b[o4]*b[o4])/4;

    w0 -= 2*(w = -4*v1*sv);
    t  -= (u[o2]+u[o4])*w;         /* u[i,j-1,k] + u[i,j+1,k] */
    w0 -= 2*(w = v1*v1);
    t  -= (u[o1]+u[o5])*w;         /* u[i,j-2,k] + u[i,j+2,k] */
 
    o1  = BOUND(i-2,d[0])-i;
    o5  = BOUND(i+2,d[0])-i;
    w0 -= 2*(w = v0*v0);
    t  -= (u[ii+o1] + u[ii+o5])*w; /* u[i-2,j,k] + u[i-2,j,k] */

    o1  = (BOUND(k-2,d[2])-k)*d[0]*d[1];
    o5  = (BOUND(k+2,d[2])-k)*d[0]*d[1];
    w0 -= 2*(w = v2*v2);
    t  -= (u[ii+o1] + u[ii+o5])*w; /* u[i,j,k-2] + u[i,j,k+2] */

    o1  = BOUND(i-1,d[0])-i;
    o5  = BOUND(i+1,d[0])-i;
    w0 -= 4*(w = 2*v0*v1);
    t  -= (u[o2+o1] + u[o2+o5] + u[o4+o1] + u[o4+o5])*w; /* u[i-1,j-1,k] + u[i+1,j-1,k] + u[i-1,j+1,k] + u[i+1,j+1,k] */
    w0 -= 2*(w = -4*v0*sv);
    t  -= (u[ii+o1] + u[ii+o5])*w; /* u[i-1,j,k] + u[i+1,j,k] */

    o1  = (BOUND(k-1,d[2])-k)*d[0]*d[1];
    o5  = (BOUND(k+1,d[2])-k)*d[0]*d[1];
    w0 -= 4*(w = 2*v1*v2);
    t  -= (u[o2+o1] + u[o2+o5] + u[o4+o1] + u[o4+o5])*w; /* u[i,j-1,k-1] + u[i,j-1,k+1] + u[i,j+1,k-1] + u[i,j+1,k+1] */
    o2  = ii+o1;
    o4  = ii+o5;
    w0 -= 2*(w = -4*v2*sv);
    t  -= (u[o2] + u[o4])*w;       /* u[i,j,k-1] + u[i,j,k+1] */

    o1 = BOUND(i-1,d[0])-i;
    o5 = BOUND(i+1,d[0])-i;
    w0 -= 4*(w = 2*v0*v2);
    t  -= (u[o2+o1] + u[o2+o5] + u[o4+o1] + u[o4+o5])*w; /* u[i-1,j,k-1] + u[i+1,j,k-1] + u[i-1,j,k+1] + u[i+1,j,k+1] */

    u[ii] = t/w0;
}

__device__ float hu_dev(USIZE_t i, USIZE_t j, USIZE_t k, const USIZE_t *d,
                       const float *u, const float *a, const float *b, const float *s)
{
    float t, v0 = s[0]*s[0], v1 = s[1]*s[1], v2 = s[2]*s[2], w, w0, sv = v0+v1+v2;
    SSIZE_t ii = i + d[0]*(j + d[1]*k), o1, o2, o4, o5;
    SSIZE_t t_;

    o1 = ii+(BOUND(j-2,d[1])-j)*d[0];
    o2 = ii+(BOUND(j-1,d[1])-j)*d[0];
    o4 = ii+(BOUND(j+1,d[1])-j)*d[0];
    o5 = ii+(BOUND(j+2,d[1])-j)*d[0];

    t  = (u[o2]*(b[o2]*a[o2] - b[ii]*a[ii]) - u[o4]*(b[o4]*a[o4] - b[ii]*a[ii])
       + (u[o1]*(b[o2]*b[o2])               + u[o5]*(b[o4]*b[o4]))/2)/2;
    w0  = a[ii]*a[ii] + (b[o2]*b[o2] + b[o4]*b[o4])/4;

    w0 -= 2*(w = -4*v1*sv);
    t  += (u[o2]+u[o4])*w;         /* u[i,j-1,k] + u[i,j+1,k] */
    w0 -= 2*(w = v1*v1);
    t  += (u[o1]+u[o5])*w;         /* u[i,j-2,k] + u[i,j+2,k] */

    o1  = BOUND(i-2,d[0])-i;
    o5  = BOUND(i+2,d[0])-i;
    w0 -= 2*(w = v0*v0);
    t  += (u[ii+o1] + u[ii+o5])*w; /* u[i-2,j,k] + u[i-2,j,k] */

    o1  = (BOUND(k-2,d[2])-k)*d[0]*d[1];
    o5  = (BOUND(k+2,d[2])-k)*d[0]*d[1];
    w0 -= 2*(w = v2*v2);
    t  += (u[ii+o1] + u[ii+o5])*w; /* u[i,j,k-2] + u[i,j,k+2] */

    o1  = BOUND(i-1,d[0])-i;
    o5  = BOUND(i+1,d[0])-i;
    w0 -= 4*(w = 2*v0*v1);
    t  += (u[o2+o1] + u[o2+o5] + u[o4+o1] + u[o4+o5])*w; /* u[i-1,j-1,k] + u[i+1,j-1,k] + u[i-1,j+1,k] + u[i+1,j+1,k] */
    w0 -= 2*(w = -4*v0*sv);
    t  += (u[ii+o1] + u[ii+o5])*w; /* u[i-1,j,k] + u[i+1,j,k] */

    o1  = (BOUND(k-1,d[2])-k)*d[0]*d[1];
    o5  = (BOUND(k+1,d[2])-k)*d[0]*d[1];
    w0 -= 4*(w = 2*v1*v2);
    t  += (u[o2+o1] + u[o2+o5] + u[o4+o1] + u[o4+o5])*w; /* u[i,j-1,k-1] + u[i,j-1,k+1] + u[i,j+1,k-1] + u[i,j+1,k+1] */
    o2  = ii+o1;
    o4  = ii+o5;
    w0 -= 2*(w = -4*v2*sv);
    t  += (u[o2] + u[o4])*w;       /* u[i,j,k-1] + u[i,j,k+1] */

    o1 = BOUND(i-1,d[0])-i;
    o5 = BOUND(i+1,d[0])-i;
    w0 -= 4*(w = 2*v0*v2);
    t  += (u[o2+o1] + u[o2+o5] + u[o4+o1] + u[o4+o5])*w; /* u[i-1,j,k-1] + u[i+1,j,k-1] + u[i-1,j,k+1] + u[i+1,j,k+1] */

    t  += u[ii]*w0;
    return(t);
}

