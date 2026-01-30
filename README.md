# PushPull

A library of [Julia](https://julialang.org/) functions that could be useful for 3D image registration. Note that code development is in its early stages and that many things are likely to change.

This package is not yet registered as an official Julia package, but has been included among a local registry of packages used by SPM.
If you are brave enouth to try using it, then the following may work (in Julia) for 64 bit Linux or Windows. First of all, you need to point Julia to the local registries used by SPM, with:
```
using Pkg
pkg"registry add https://github.com/spm/SPM-registry.jl"
```
The above only needs to be done once. You can then install using:
```
using Pkg
Pkg.add("PushPull.jl")
```


[![Build Status](https://github.com/spm/PushPull.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/spm/PushPull.jl/actions/workflows/CI.yml?query=branch%3Amain)

