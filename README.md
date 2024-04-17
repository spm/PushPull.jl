# PushPull

A library of [Julia](https://julialang.org/) functions that could be useful for 3D image registration. Note that code development is in its early stages and that many things are likely to change.

This package is not yet registered as an official Julia package.
If you are brave enouth to try using it, then the following may work (in Julia) for 64 bit Linux or Windows:

    using Pkg
    Pkg.develop(url="https://github.com/spm/PushPull.jl")
    using PushPull


Note that the automated githib actions fail because CUDA drivers are missing, which leads on to several other problems.
Multi-dimensionsional ffts on GPU (CUFFT) can also be [problematic](https://github.com/JuliaGPU/CUDA.jl/issues/119) with older Julia versions (fixed somewhere between Julia 1.7.2 and 1.9.3).

[![Build Status](https://github.com/spm/PushPull.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/spm/PushPull.jl/actions/workflows/CI.yml?query=branch%3Amain)

