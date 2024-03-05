using Libdl

"""
    pull(f₀::Array{Float32}, ϕ::Array{Float32}, sett::Settings)

Pull 3D volume(s) `f₀` using transform `ϕ` into a new image.
This operation is the adjoint of `push`.

Requirements
* `f₀` must have between 3 & 5 dimensions
* `ϕ` must have 4 or 5 dimensions
* `ϕ` must have three channels
* `ϕ` & `f₀` must have the same batch size

"""
function pull(f₀::Array{Float32}, ϕ::Array{Float32}, sett::Settings = Settings())

    global pplib

    @assert(ndims(f₀) >= 3 && ndims(f₀) <= 5, "`f₀` must have between 3 & 5 dimensions")
    @assert(ndims(ϕ)  >= 4 && ndims(ϕ)  <= 5, "`ϕ` must have 4 or 5 dimensions")
    @assert(size(ϕ,4) == 3,                   "`ϕ` must have three channels")
    @assert(size(ϕ,5) == size(f₀,5),          "`ϕ` & `f₀` must have the same batch size")

    Nc  = size(f₀,4)          # Number of channels
    Nb  = size(f₀,5)          # Batchsize
    dv  = (tmp=size(f₀); tmp[4:end]) 
    n₀  = prod(size(f₀)[1:3]) # Original volume dimensions
    d₁  = size(ϕ)[1:3]        # Output volume dimensions
    n₁  = prod(d₁)            # Number of voxels in output volume

    dp  = Csize_t.([sett.deg...].+Csize_t(1))
    bnd = Cint.([sett.bnd...])
    ext = sett.ext
    d₀  = Csize_t.([size(f₀)[1:3]...])
    n₁  = Csize_t.(n₁)

    f₁  = zeros(Float32, (d₁..., dv...))

    for nb=1:Nb, nc=1:Nc
        ccall(dlsym(pplib,:pull), Cvoid,
              (Ref{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat},
               Ptr{Csize_t}, Csize_t, Ptr{Cint}, Ptr{Csize_t}, Cint),
               pointer(f₁, 1 + n₁*(Nc*(nb-1) + nc-1)), pointer(ϕ, 1 + 3n₁*(nb-1)), pointer(f₀, 1 + n₀*(Nc*(nb-1) + nc-1)),
               pointer(d₀), n₁, pointer(bnd), pointer(dp), Cint(sett.ext))
    end
    return f₁
end


"""
    pull_grad(f₀::Array{Float32}, ϕ::Array{Float32}, sett::Settings)

Pull gradients of 3D volume(s) `f₀` using transform `ϕ`.
This operation is the adjoint of `push_grad`.

Requirements
* `f₀` must have between 3 & 5 dimensions
* `ϕ` must have 4 or 5 dimensions
* `ϕ` must have three channels
* `ϕ` & `f₀` must have the same batch size

"""
function pull_grad(f₀::Array{Float32}, ϕ::Array{Float32}, sett::Settings = Settings())

    global pplib

    @assert(ndims(f₀) >= 3 && ndims(f₀) <= 5, "`f₀` must have between 3 & 5 dimensions")
    @assert(ndims(ϕ)  >= 4 && ndims(ϕ)  <= 5, "`ϕ` must have 4 or 5 dimensions")
    @assert(size(ϕ,4) == 3,                   "`ϕ` must have three channels")
    @assert(size(ϕ,5) == size(f₀,5),          "`ϕ` & `f₀` must have the same batch size")

    Nc  = size(f₀,4)          # Number of channels
    Nb  = size(f₀,5)          # Batchsize
    dv  = (tmp=size(f₀); tmp[4:end])
    n₀  = prod(size(f₀)[1:3]) # Original volume dimensions
    d₁  = size(ϕ)[1:3]        # Output volume dimensions
    n₁  = prod(d₁)            # Number of voxels in output volume

    dp  = Csize_t.([sett.deg...].+Csize_t(1))
    bnd = Cint.([sett.bnd...])
    ext = sett.ext
    d₀  = Csize_t.([size(f₀)[1:3]...])
    n₁  = Csize_t.(n₁)

    ∇f  = zeros(Float32, (d₁..., 3, dv...))

    for nb=1:Nb, nc=1:Nc
        # void pull_grad(float *∇f, const float *ϕ, const float *f₀,
        #                const USIZE_t *d₀, const USIZE_t n₁, const int *bnd, const USIZE_t *dp, const int ext)

        ccall(dlsym(pplib,:pullg), Cvoid,
              (Ref{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat},
               Ptr{Csize_t}, Csize_t, Ptr{Cint}, Ptr{Csize_t}, Cint),
               pointer(∇f, 1 + 3n₁*(Nc*(nb-1) + nc-1)), pointer(ϕ, 1 + 3n₁*(nb-1)), pointer(f₀, 1 + n₀*(Nc*(nb-1) + nc-1)),
               pointer(d₀), n₁, pointer(bnd), pointer(dp), Cint(sett.ext))
    end
    return ∇f
end


function pull_hess(f₀::Array{Float32}, ϕ::Array{Float32}, sett::Settings = Settings())

    global pplib

    @assert(ndims(f₀) >= 3 && ndims(f₀) <= 5, "`f₀` must have between 3 & 5 dimensions")
    @assert(ndims(ϕ)  >= 4 && ndims(ϕ)  <= 5, "`ϕ` must have 4 or 5 dimensions")
    @assert(size(ϕ,4) == 3,                   "`ϕ` must have three channels")
    @assert(size(ϕ,5) == size(f₀,5),          "`ϕ` & `f₀` must have the same batch size")

    Nc  = size(f₀,4)          # Number of channels
    Nb  = size(f₀,5)          # Batchsize
    dv  = (tmp=size(f₀); tmp[4:end])
    n₀  = prod(size(f₀)[1:3]) # Original volume dimensions
    d₁  = size(ϕ)[1:3]        # Output volume dimensions
    n₁  = prod(d₁)            # Number of voxels in output volume

    dp  = Csize_t.([sett.deg...].+Csize_t(1))
    bnd = Cint.([sett.bnd...])
    ext = sett.ext
    d₀  = Csize_t.([size(f₀)[1:3]...])
    n₁  = Csize_t.(n₁)

    h₁  = zeros(Float32, (d₁..., 3,3, dv...))

    for nb=1:Nb, nc=1:Nc
        ccall(dlsym(pplib,:pullh), Cvoid,
              (Ref{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat},
              Ptr{Csize_t}, Csize_t, Ptr{Cint}, Ptr{Csize_t}, Cint),
              pointer(h₁, 1 + 9n₁*(Nc*(nb-1) + nc-1)), pointer(ϕ, 1 + 3n₁*(nb-1)), pointer(f₀, 1 + n₀*(Nc*(nb-1) + nc-1)),
              pointer(d₀), n₁, pointer(bnd), pointer(dp), Cint(sett.ext))
    end
    return h₁
end


"""
    push(f₁::Array{Float32}, ϕ::Array{Float32}, d₁::NTuple{3,Integer}, sett::Settings)

Push 3D volume(s) `f₁` using transform `ϕ` into a new image of dimensions `d₁`.
This operation is the adjoint of `pull`.

Requirements
* `f₁` must have between 3 & 5 dimensions
* `ϕ` must have 4 or 5 dimensions
* `ϕ` must have three channels
* `ϕ` & `f₀` must have the same batch size
* `f₁` and `ϕ` must have the same volume dimensions

"""
function push(f₁::Array{Float32}, ϕ::Array{Float32}, d₀::NTuple{3,Integer}, sett::Settings = Settings())

    global pplib

    @assert(ndims(f₁) >= 3 && ndims(f₁) <= 5,   "`f₁` must have between 3 & 5 dimensions")
    @assert(ndims(ϕ)  >= 4 && ndims(ϕ)  <= 5,   "`ϕ` must have 4 or 5 dimensions")
    @assert(size(ϕ,4) == 3,                     "`ϕ` must have three channels")
    @assert(size(ϕ,5) == size(f₁,5),            "`ϕ` & `f₀` must have the same batch size")
    @assert(all(size(f₁)[1:3] == size(ϕ)[1:3]), "`f₁` and `ϕ` must have the same volume dimensions")

    Nc  = size(f₁,4)          # Number of channels
    Nb  = size(f₁,5)          # Batchsize
    dv  = (tmp=size(f₁); tmp[4:end])
    n₀  = prod(d₀)            # Output volume dimensions
    d₁  = size(ϕ)[1:3]        # Input volume dimensions
    n₁  = prod(d₁)            # Number of voxels in input volume

    dp  = Csize_t.([sett.deg...].+Csize_t(1))
    bnd = Cint.([sett.bnd...])
    ext = sett.ext
    d₀  = Csize_t.([d₀...])
    n₁  = Csize_t.(n₁)

    f₀  = zeros(Float32, (d₀..., dv...))

    for nb=1:Nb, nc=1:Nc
        ccall(dlsym(pplib,:push), Cvoid,
              (Ref{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat},
               Ptr{Csize_t}, Csize_t, Ptr{Cint}, Ptr{Csize_t}, Cint),
               pointer(f₀, 1 + n₀*(Nc*(nb-1) + nc-1)), pointer(ϕ, 1 + 3n₁*(nb-1)), pointer(f₁, 1 + n₁*(Nc*(nb-1) + nc-1)),
               pointer(d₀), n₁, pointer(bnd), pointer(dp), Cint(sett.ext))
    end
    return f₀
end


"""
    push_grad(∇f::Array{Float32}, ϕ::Array{Float32}, d₁::NTuple{3,Integer}, sett::Settings)

Push 3D gradients `∇f` using transform `ϕ` into a new image of dimensions `d₁`.
This operation is the adjoint of `pull_grad`.

Requirements
* `∇f` must have between 4 & 6 dimensions
* `ϕ` must have 4 or 5 dimensions
* `ϕ` must have three channels
* `∇f` must have three components
* `∇f` and `ϕ` must have the same volume dimensions

"""
function push_grad(∇f::Array{Float32}, ϕ::Array{Float32}, d₀::NTuple{3,Integer}, sett::Settings = Settings())

    global pplib

    @assert(ndims(∇f) >= 4 && ndims(∇f) <= 6,   "`∇f` must have between 4 & 6 dimensions")
    @assert(ndims(ϕ)  >= 4 && ndims(ϕ)  <= 5,   "`ϕ` must have 4 or 5 dimensions")
    @assert(size(ϕ, 4) == 3,                    "`ϕ` must have three channels")
    @assert(size(∇f,4) == 3,                    "`∇f` must have three components")
    @assert(size(ϕ, 5) == size(∇f,6),           "`ϕ` & `∇f` must have the same batch size")
    @assert(all(size(∇f)[1:3] == size(ϕ)[1:3]), "`∇f` and `ϕ` must have the same volume dimensions")

    Nc  = size(∇f,5)          # Number of channels
    Nb  = size(∇f,6)          # Batchsize
    dv  = (tmp=size(∇f); tmp[5:end])
    n₀  = prod(d₀)            # Output volume dimensions
    d₁  = size(ϕ)[1:3]        # Input volume dimensions
    n₁  = prod(d₁)            # Number of voxels in input volume

    dp  = Csize_t.([sett.deg...].+Csize_t(1))
    bnd = Cint.([sett.bnd...])
    ext = sett.ext
    d₀  = Csize_t.([d₀...])
    n₁  = Csize_t.(n₁)

    g₀  = zeros(Float32, (d₀..., dv...))

    for nb=1:Nb, nc=1:Nc
        ccall(dlsym(pplib,:pushg), Cvoid,
              (Ref{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat},
              Ptr{Csize_t}, Csize_t, Ptr{Cint}, Ptr{Csize_t}, Cint),
              pointer(g₀, 1 + n₀*(Nc*(nb-1) + nc-1)), pointer(ϕ, 1 + 3n₁*(nb-1)), pointer(∇f, 1 + 3n₁*(Nc*(nb-1) + nc-1)),
              pointer(d₀), n₁, pointer(bnd), pointer(dp), Cint(sett.ext))
    end
    return g₀
end


"""
    affine_pull(f₀::Array{Float32}, Aff::Array{Float32,2}, d₁::NTuple{3,Integer}, sett::Settings = Settings())

This function is still work in progress.

"""
function affine_pull(f₀::Array{Float32}, Aff::Array{Float32,2}, d₁::NTuple{3,Integer}, sett::Settings = Settings())

    A = adjust_affine(Aff)

    @assert(ndims(f₀) >= 3 && ndims(f₀) <= 5, "`f₀` must have between 3 & 5 dimensions")
    Nc  = size(f₀,4)          # Number of channels
    Nb  = size(f₀,5)          # Batchsize
    dv  = size(f₀)[4:end]
    d₀  = size(f₀)[1:3]
    d₀  = Csize_t.([d₀...])
    n₀  = prod(d₀)            # Original volume dimensions
    d₁  = Csize_t.([d₁...])
    n₁  = prod(d₁)            # Number of voxels in output volume

    dp  = Csize_t.([sett.deg...].+Csize_t(1))
    bnd = Cint.([sett.bnd...])
    ext = sett.ext

    f₁  = zeros(Float32, (d₁..., dv...))

    for nb=1:Nb, nc=1:Nc
        ccall(dlsym(pplib,:pull_affine), Cvoid,
              (Ref{Cfloat},  Ptr{Cfloat},
               Ptr{Csize_t}, Csize_t,
               Ptr{Csize_t}, Ptr{Cfloat},
               Ptr{Cint},    Ptr{Csize_t}, Cint),
              pointer(f₁, 1 + n₁*(Nc*(nb-1) + nc-1)), pointer(f₀, 1 + n₀*(Nc*(nb-1) + nc-1)),
              pointer(d₀), n₁,
              pointer(d₁), pointer(A), 
              pointer(bnd), pointer(dp), Cint(sett.ext))
    end

    return f₁
end


function affine_push(f₁::Array{Float32}, Aff::Array{Float32,2}, d₀::NTuple{3,Integer}, sett::Settings = Settings())

    global pplib

    A = adjust_affine(Aff)

    @assert(ndims(f₁) >= 3 && ndims(f₁) <= 5,   "`f₁` must have between 3 & 5 dimensions")
    Nc  = size(f₁,4)          # Number of channels
    Nb  = size(f₁,5)          # Batchsize
    dv  = size(f₁)[4:end]
    d₁  = size(f₁)[1:3]       # Input volume dimensions
    d₁  = Csize_t.([d₁...])
    n₁  = prod(d₁)            # Number of voxels in input volume
    d₀  = Csize_t.([d₀...])
    n₀  = prod(d₀)            # Output volume dimensions

    dp  = Csize_t.([sett.deg...].+Csize_t(1))
    bnd = Cint.([sett.bnd...])
    ext = sett.ext

    f₀  = zeros(Float32, (d₀..., dv...))

    for nb=1:Nb, nc=1:Nc
        ccall(dlsym(pplib,:push_affine), Cvoid,
              (Ref{Cfloat}, Ptr{Cfloat},
               Ptr{Csize_t}, Csize_t,
               Ptr{Csize_t}, Ptr{Cfloat},
               Ptr{Cint}, Ptr{Csize_t}, Cint),
              pointer(f₀, 1 + n₀*(Nc*(nb-1) + nc-1)), pointer(f₁, 1 + n₁*(Nc*(nb-1) + nc-1)),
              pointer(d₀), n₁,
              pointer(d₁), pointer(A),
              pointer(bnd), pointer(dp), Cint(sett.ext))
    end
    return f₀
end

"""
    A = adjust_affine(Aff::Array{Float32,2})

Adjust affine transform matrix from 1-offset to 0-offset.

"""
function adjust_affine(Aff::Array{Float32,2})
    @assert((size(Aff,1)==3 || size(Aff,1)==4) && size(Aff,2)==4)
    A = Float32.(Aff[1:3,:])        # Assume A[4,:]==[0 0 0 1]
    A[:,4] .= sum(A,dims=2) .- 1.0  # Adjust for 0-offset (C code)
    return A
end

