
"""
    pull(f₀::CuArray{Float32}, ϕ::CuArray{Float32}, sett::Settings)

Pull 3D volume `f₀` using transform `ϕ` into a new image. This operation
is the adjoint of `push`.

Requirements
* `f₀` must have between 3 & 5 dimensions
* `ϕ` must have 4 or 5 dimensions
* `ϕ` must have three channels
* `ϕ` & `f₀` must have the same batch size

"""
function PushPull.pull(f₀::CuArray{Float32}, ϕ::CuArray{Float32}, sett::Settings = Settings())::CuArray{Float32}

    ppmod  = getppmod()
    cuPull = CuFunction(ppmod, "_Z12pull_elementPfPKfS1_")

    @assert(ndims(f₀) >= 3 && ndims(f₀) <= 5, "`f₀` must have between 3 & 5 dimensions")
    @assert(ndims(ϕ)  >= 4 && ndims(ϕ)  <= 5, "`ϕ` must have 4 or 5 dimensions")
    @assert(size(ϕ,4) == 3,                   "`ϕ` must have three channels")
    @assert(size(ϕ,5) == size(f₀,5),          "`ϕ` & `f₀` must have the same batch size")
    @assert(length(sett.bnd)==size(f₀,4) || length(sett.bnd)==1)

    Nc  = size(f₀,4)          # Number of channels
    Nb  = size(f₀,5)          # Batchsize
    dv  = (tmp=size(f₀); tmp[4:end])
    n₀  = prod(size(f₀)[1:3]) # Original volume dimensions
    d₁  = size(ϕ)[1:3]        # Output volume dimensions
    n₁  = prod(d₁)            # Number of voxels in output volume

    gpusettings(size(f₀)[1:3], n₁, sett)

    f₁  = CUDA.zeros(Float32, (d₁..., dv...))

    threads,blocks = threadblocks(cuPull,n₁)
    for nb=1:Nb, nc=1:Nc
        setbound(nc, sett)
        cudacall(cuPull, (CuPtr{Cfloat},CuPtr{Cfloat},CuPtr{Cfloat}),
                 pointer(f₁, 1 + n₁*(Nc*(nb-1) + nc-1)),
                 pointer(ϕ, 1 + 3n₁*(nb-1)), pointer(f₀, 1 + n₀*(Nc*(nb-1) + nc-1));
                 threads=threads, blocks=blocks)
    end
    return f₁
end


"""
    pull_grad(f₀::CuArray{Float32}, ϕ::CuArray{Float32}, sett::Settings)

Pull gradients of 3D volume `f₀` using transform `ϕ`.

Requirements
* `f₀` must have between 3 & 5 dimensions
* `ϕ` must have 4 or 5 dimensions
* `ϕ` must have three channels
* `ϕ` & `f₀` must have the same batch size

"""
function PushPull.pull_grad(f₀::CuArray{Float32}, ϕ::CuArray{Float32}, sett::Settings = Settings())::CuArray{Float32}

    ppmod = getppmod()
    cuPullGrad = CuFunction(ppmod, "_Z13pullg_elementPfPKfS1_")

    @assert(ndims(f₀) >= 3 && ndims(f₀) <= 5, "`f₀` must have between 3 & 5 dimensions")
    @assert(ndims(ϕ)  >= 4 && ndims(ϕ)  <= 5, "`ϕ` must have 4 or 5 dimensions")
    @assert(size(ϕ,4) == 3,                   "`ϕ` must have three channels")
    @assert(size(ϕ,5) == size(f₀,5),          "`ϕ` & `f₀` must have the same batch size")
    @assert(length(sett.bnd)==size(f₀,4) || length(sett.bnd)==1)

    Nc  = size(f₀,4)          # Number of channels
    Nb  = size(f₀,5)          # Batchsize
    dv  = (tmp=size(f₀); tmp[4:end])
    n₀  = prod(size(f₀)[1:3]) # Original volume dimensions
    d₁  = size(ϕ)[1:3]        # Output volume dimensions
    n₁  = prod(d₁)            # Number of voxels in output volume

    gpusettings(size(f₀)[1:3], n₁, sett)

    ∇f  = CUDA.zeros(Float32, (d₁..., 3, dv...))

    threads,blocks = threadblocks(cuPullGrad,n₁)
    for nb=1:Nb, nc=1:Nc
        setbound(nc, sett)
        cudacall(cuPullGrad, (CuPtr{Cfloat},CuPtr{Cfloat},CuPtr{Cfloat}),
                 pointer(∇f, 1 + 3n₁*(Nc*(nb-1) + nc-1)), pointer(ϕ, 1 + 3n₁*(nb-1)), pointer(f₀, 1 + n₀*(Nc*(nb-1) + nc-1));
                 threads=threads, blocks=blocks)
    end
    return ∇f
end

"""
    pull_hess(f₀::CuArray{Float32}, ϕ::CuArray{Float32}, sett::Settings)

Pull hessian of 3D volume `f₀` using transform `ϕ`.

Requirements
* `f₀` must have between 3 & 5 dimensions
* `ϕ` must have 4 or 5 dimensions
* `ϕ` must have three channels
* `ϕ` & `f₀` must have the same batch size

"""
function PushPull.pull_hess(f₀::CuArray{Float32}, ϕ::CuArray{Float32}, sett::Settings = Settings())::CuArray{Float32}

    ppmod      = getppmod()
    cuPullHess = CuFunction(ppmod, "_Z13pullh_elementPfPKfS1_")

    @assert(ndims(f₀) >= 3 && ndims(f₀) <= 5, "`f₀` must have between 3 & 5 dimensions")
    @assert(ndims(ϕ)  >= 4 && ndims(ϕ)  <= 5, "`ϕ` must have 4 or 5 dimensions")
    @assert(size(ϕ,4) == 3,                   "`ϕ` must have three channels")
    @assert(size(ϕ,5) == size(f₀,5),          "`ϕ` & `f₀` must have the same batch size")
    @assert(length(sett.bnd)==size(f₀,4) || length(sett.bnd)==1)

    Nc  = size(f₀,4)          # Number of channels
    Nb  = size(f₀,5)          # Batchsize
    dv  = (tmp=size(f₀); tmp[4:end])
    n₀  = prod(size(f₀)[1:3]) # Original volume dimensions
    d₁  = size(ϕ)[1:3]        # Output volume dimensions
    n₁  = prod(d₁)            # Number of voxels in output volume

    gpusettings(size(f₀)[1:3], n₁, sett)

    h₁  = CUDA.zeros(Float32, (d₁..., 3,3, dv...))

    threads,blocks = threadblocks(cuPullHess,n₁)
    for nb=1:Nb, nc=1:Nc
        setbound(nc, sett)
        cudacall(cuPullHess, (CuPtr{Cfloat},CuPtr{Cfloat},CuPtr{Cfloat}),
                 pointer(h₁, 1 + 9n₁*(Nc*(nb-1) + nc-1)), pointer(ϕ, 1 + 3n₁*(nb-1)), pointer(f₀, 1 + n₀*(Nc*(nb-1) + nc-1));
                 threads=threads, blocks=blocks)
    end
    return h₁
end


"""
    push(f₁::CuArray{Float32}, ϕ::CuArray{Float32,4}, d₁::NTuple{3,Integer}, sett::Settings)

Push 3D volume `f₁` using transform `ϕ` into a new image of dimensions `d₁`. This
operation is the adjoint of `pull`.

Requirements
* `f₁` must have between 3 & 5 dimensions
* `ϕ` must have 4 or 5 dimensions
* `ϕ` & `f₀` must have the same batch size
* `ϕ` & `f₀` must have the same batch size

"""
function PushPull.push(f₁::CuArray{Float32}, ϕ::CuArray{Float32}, d₀::NTuple{3,Integer},
              sett::Settings = Settings())::CuArray{Float32}

    ppmod  = getppmod()
    cuPush = CuFunction(ppmod, "_Z12push_elementPfPKfS1_")

    @assert(ndims(f₁) >= 3 && ndims(f₁) <= 5,   "`f₁` must have between 3 & 5 dimensions")
    @assert(ndims(ϕ)  >= 4 && ndims(ϕ)  <= 5,   "`ϕ` must have 4 or 5 dimensions")
    @assert(size(ϕ,4) == 3,                     "`ϕ` must have three channels")
    @assert(size(ϕ,5) == size(f₁,5),            "`ϕ` & `f₀` must have the same batch size")
    @assert(all(size(f₁)[1:3] == size(ϕ)[1:3]), "`f₁` and `ϕ` must have the same volume dimensions")
    @assert(length(sett.bnd)==size(f₁,4) || length(sett.bnd)==1)

    Nc  = size(f₁,4)          # Number of channels
    Nb  = size(f₁,5)          # Batchsize
    dv  = (tmp=size(f₁); tmp[4:end])
    n₀  = prod(d₀)            # Output volume dimensions
    d₁  = size(ϕ)[1:3]        # Input volume dimensions
    n₁  = prod(d₁)            # Number of voxels in input volume

    gpusettings(d₀, n₁, sett)

    f₀  = CUDA.zeros(Float32, (d₀..., dv...))

    threads,blocks = threadblocks(cuPush,n₁)
    for nb=1:Nb, nc=1:Nc
        setbound(nc, sett)
        cudacall(cuPush, (CuPtr{Cfloat},CuPtr{Cfloat},CuPtr{Cfloat}),
                 pointer(f₀, 1 + n₀*(Nc*(nb-1) + nc-1)), pointer(ϕ, 1 + 3n₁*(nb-1)), pointer(f₁, 1 + n₁*(Nc*(nb-1) + nc-1));
                 threads=threads, blocks=blocks)
    end
    return f₀
end


"""
    push_grad(f₁::CuArray{Float32}, ϕ::CuArray{Float32,4}, d₁::NTuple{3,Integer}, sett::Settings)

Push 3D volume `∇f` gradients using transform `ϕ` into a new image of dimensions `d₁`. This
operation is the adjoint of `pull_grad`.

Requirements
* `∇f` must have between 4 & 6 dimensions
* `ϕ` must have 4 or 5 dimensions
* `∇f` must have three components
* `ϕ` & `∇f` must have the same batch size
* `∇f` and `ϕ` must have the same volume dimensions

"""
function PushPull.push_grad(∇f::CuArray{Float32}, ϕ::CuArray{Float32}, d₀::NTuple{3,Integer},
                   sett::Settings = Settings())::CuArray{Float32}

    ppmod      = getppmod()
    cuPushGrad = CuFunction(ppmod, "_Z13pushg_elementPfPKfS1_")

    @assert(ndims(∇f) >= 4 && ndims(∇f) <= 6,   "`∇f` must have between 4 & 6 dimensions")
    @assert(ndims(ϕ)  >= 4 && ndims(ϕ)  <= 5,   "`ϕ` must have 4 or 5 dimensions")
    @assert(size(ϕ, 4) == 3,                    "`ϕ` must have three channels")
    @assert(size(∇f,4) == 3,                    "`∇f` must have three components")
    @assert(size(ϕ, 5) == size(∇f,6),           "`ϕ` & `∇f` must have the same batch size")
    @assert(all(size(∇f)[1:3] == size(ϕ)[1:3]), "`∇f` and `ϕ` must have the same volume dimensions")
    @assert(length(sett.bnd)==size(∇f,5) || length(sett.bnd)==1)

    Nc  = size(∇f,5)          # Number of channels
    Nb  = size(∇f,6)          # Batchsize
    dv  = (tmp=size(∇f); tmp[5:end])
    n₀  = prod(d₀)            # Output volume dimensions
    d₁  = size(ϕ)[1:3]        # Input volume dimensions
    n₁  = prod(d₁)            # Number of voxels in input volume

    gpusettings(d₀, n₁, sett)

    g₀  = CUDA.zeros(Float32, (d₀..., dv...))

    threads,blocks = threadblocks(cuPushGrad,n₁)
    for nb=1:Nb, nc=1:Nc
        setbound(nc, sett)
        cudacall(cuPushGrad, (CuPtr{Cfloat},CuPtr{Cfloat},CuPtr{Cfloat}),
                 pointer(g₀, 1 + n₀*(Nc*(nb-1) + nc-1)), pointer(ϕ, 1 + 3n₁*(nb-1)), pointer(∇f, 1 + 3n₁*(Nc*(nb-1) + nc-1));
                 threads=threads, blocks=blocks)
    end
    return g₀
end


"""
    gpusettings(d₀, n₁, sett::Settings = Settings())

Put interpolation settings into global variables on GPU.

"""
function gpusettings(d₀, n₁, sett::Settings)
    ppmod = getppmod()
    setindex!(CuGlobal{NTuple{3,Csize_t}}(ppmod,"dp"),  Csize_t.(sett.deg).+Csize_t(1))
    setindex!(CuGlobal{NTuple{3, Int32}}(ppmod,"bnd"),  sett.bnd[1]) # Might need much more work
    setindex!(CuGlobal{Int32}(ppmod,"ext"),             sett.ext)

    setindex!(CuGlobal{NTuple{3,Csize_t}}(ppmod,"d0"),  Csize_t.(d₀[1:3]))
    setindex!(CuGlobal{Csize_t}(ppmod,"n1"),            Csize_t(n₁))
    nothing
end

"""
affine_pull(f₀::CuArray{Float32}, Aff::Array{Float32,2}, d₁::NTuple{3,Integer}, sett::Settings = Settings())

Work in progress

"""
function PushPull.affine_pull(f₀::CuArray{Float32}, Aff::Array{Float32,2}, d₁::NTuple{3,Integer},
                     sett::Settings = Settings())::CuArray{Float32}

    ppmod     = getppmod()
    cuAffPull = CuFunction(ppmod, "_Z19affine_pull_elementPfPKf")

    @assert((size(Aff,1)==3 || size(Aff,1)==4) && size(Aff,2)==4)
    A = Float32.(Aff[1:3,1:4])

    @assert(ndims(f₀) >= 3 && ndims(f₀) <= 5, "`f₀` must have between 3 & 5 dimensions")
    @assert(length(sett.bnd)==size(f₀,4) || length(sett.bnd)==1)

    Nc  = size(f₀,4)          # Number of channels
    Nb  = size(f₀,5)          # Batchsize
    dv  = size(f₀)[4:end]
    d₀  = size(f₀)[1:3]
    d₀  = Csize_t.(d₀)
    n₀  = prod(d₀)            # Original volume dimensions
    d₁  = Csize_t.(d₁)
    n₁  = prod(d₁)            # Number of voxels in output volume

    gpusettings(d₀, n₁, sett)
    gpusettings_aff(d₁, A)

    f₁  = CUDA.zeros(Float32, (d₁..., dv...))

    threads,blocks = threadblocks(cuAffPull,n₁)
    for nb=1:Nb, nc=1:Nc
        setbound(nc, sett)
        cudacall(cuAffPull, (CuPtr{Cfloat}, CuPtr{Cfloat}),
                 pointer(f₁,1 + n₁*(Nc*(nb-1) + nc-1)), pointer(f₀, 1 + n₀*(Nc*(nb-1) + nc-1));
                 threads=threads, blocks=blocks)
    end
    return f₁
end

"""
affine_push(f₁::CuArray{Float32}, Aff::Array{Float32,2}, d₀::NTuple{3,Integer}, sett::Settings = Settings())

Work in progress

"""
function PushPull.affine_push(f₁::CuArray{Float32}, Aff::Array{Float32,2}, d₀::NTuple{3,Integer},
                     sett::Settings = Settings())::CuArray{Float32}

    ppmod      = getppmod()
    cuAffPush  = CuFunction(ppmod, "_Z19affine_push_elementPfPKf")

    @assert((size(Aff,1)==3 || size(Aff,1)==4) && size(Aff,2)==4)
    A = Float32.(Aff[1:3,1:4])

    @assert(ndims(f₁) >= 3 && ndims(f₁) <= 5,   "`f₁` must have between 3 & 5 dimensions")
    @assert(length(sett.bnd)==size(f₁,4) || length(sett.bnd)==1)

    Nc  = size(f₁,4)          # Number of channels
    Nb  = size(f₁,5)          # Batchsize
    dv  = size(f₁)[4:end]
    n₀  = prod(d₀)            # Output volume dimensions
    d₁  = size(f₁)[1:3]       # Input volume dimensions
    n₁  = prod(d₁)            # Number of voxels in input volume

    gpusettings(d₀, n₁, sett)
    gpusettings_aff(d₁, A)

    f₀  = CUDA.zeros(Float32, (d₀..., dv...))

    threads,blocks = threadblocks(cuAffPush,n₁)
    for nb=1:Nb, nc=1:Nc
        setbound(nc, sett)
        cudacall(cuAffPush, (CuPtr{Cfloat},CuPtr{Cfloat}),
                 pointer(f₀, 1 + n₀*(Nc*(nb-1) + nc-1)), pointer(f₁, 1 + n₁*(Nc*(nb-1) + nc-1));
                 threads=threads, blocks=blocks)
    end
    return f₀
end


function gpusettings_aff(d₁,Aff)
    ppmod = getppmod()
    setindex!(CuGlobal{NTuple{3,Csize_t}}(ppmod,"d1"),  Csize_t.(d₁))
    Aff = Aff[1:3,:]                    # Assume Aff[4,:]==[0 0 0 1]
    Aff[:,4] .= sum(Aff,dims=2) .- 1.0  # Adjust for 0-offset (CUDA code)
    Aff = (Float32.(Aff)[:]...,)
    setindex!(CuGlobal{NTuple{12,Float32}}(ppmod,"Aff"),  Aff)
    nothing
end

function setbound(nc::Integer, sett::Settings)
    ppmod = getppmod()
    bnd = (length(sett.bnd)==1 ? sett.bnd[1] : sett.bnd[nc])
    setindex!(CuGlobal{NTuple{3, Int32}}(ppmod,"bnd"), bnd)
end

function threadblocks(fun,n)
    config  = launch_configuration(fun; max_threads=n)
    threads = config.threads
    blocks  = Int32(ceil(n./threads))
    return threads, blocks
end

function getppmod()
    ppmod = CuModuleFile(joinpath(PushPull.ptxdir(), "pushpull.ptx"))
end

