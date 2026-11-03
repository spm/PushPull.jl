/*
 * John Ashburner
 * Copyright (C) 2023 Department of Imaging Neuroscience, UCL
 */

#define SSIZE_t long long
#define USIZE_t unsigned long long

void refl_b(SSIZE_t i, USIZE_t d, /*@out@*/USIZE_t *b, /*@out@*/int *s)
{
    USIZE_t d2 = d<<1;
    USIZE_t i1 = ((USIZE_t)(i%(SSIZE_t)d2) + d2) % d2;
    *b = (i1<d) ? i1 : d2-(i1+1);
    *s = 1;
}


void refl1_b(SSIZE_t i, USIZE_t d, /*@out@*/USIZE_t *b, /*@out@*/int *s)
{
    USIZE_t d2 = d<<1;
    USIZE_t i1 = ((USIZE_t)(i%(SSIZE_t)d2) + d2) % d2;
    if(i1<d)
    {
        *b = i1;
        *s = 1;
    }
    else
    {
        *b = d2-(i1+1);
        *s = -1;
    }
}

#include<stdio.h>

int main()
{
    SSIZE_t i;
    USIZE_t d = 4, b;
    int s;

    for(i=-5; i<10; i++)
    {
        refl1_b(i, d, &b, &s);
        printf("%d %d %d\n", (int)i, (int)b, (int)s);
    }
    return(0);
}

