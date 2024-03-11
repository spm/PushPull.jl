# PushPull

A library of functions that could be useful for registering images. Note that code development is in its early stages and that many things will change.

This package is not yet reistered as an official Julia package.
If you are brave enouth to try using it, then the following may work:
    using Pkg
    Pkg.develop("https://github.com/spm/PushPull.jl")
    using PushPull

Note that the automated githib actions fail because there seems to be a problem in downloading the CUDA driver, leading on to several other problems.

[![Build Status](https://github.com/spm/PushPull.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/spm/PushPull.jl/actions/workflows/CI.yml?query=branch%3Amain)
