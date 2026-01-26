using CUDA

import Base.cumprod
cumprod(d::Tuple)=(cumprod([d...,])...,)

global tvmod


function TVdenoise(x::Union{CuArray{Float32,3},CuArray{Float32,4}}, nit::Integer=1,
                   vox::NTuple{3,Real}=(1.0f0,1.0f0,1.0f0), lambda::Union{Real,Array{Real}}=1.0f0)
    y = deepcopy(x)
    y = TVdenoise!(x,y,nit,vox,lambda)
    return y
end


function TVdenoise!(x::Union{CuArray{Float32,3},CuArray{Float32,4}},
                    y::Union{CuArray{Float32,3},CuArray{Float32,4}},
                    nit::Integer, vox::NTuple{3,Real}=(1.0f0,1.0f0,1.0f0),
                    lambdap::Union{Real,Array{Real}}=1.0f0, lambdal::Union{Real,Array{Real}}=1.0f0)

    global tvmod
    cutv3d = CuFunction(tvmod, "_Z11TVdenoise3dPfPKf")
    nlam = 20 # A constant from the .cu

    @assert(size(x)==size(y), "incompatible sizes of input and output")
    d    = UInt64.([size(x)..., 1, 1]) # Gradient dimensions
    @assert(prod(d[5:length(d)])==1,"too many dimensions")
    d    = (d[1:4]...,)
    @assert(d[4]<=nlam, "too many volumes")

    if isa(lambdap,Real)
        lambdap = Float32(lambdap).*(ones(Float32,nlam)...,)
    elseif isa(Lambda,Array) || isa(Lambda,NTuple)
        if length(Lambda)~=d[4]
            error("incompatible size of lambdap")
        end
        lambdap = (Float32.(lambdap)..., zeros(Float32,nlam-length(lambdap))...)
    elseif numel(lambdap)==1
        error("wrong type of lambdap")
    end

    if isa(lambdal,Real)
        lambdal = Float32(lambdal).*(ones(Float32,nlam)...,)
    elseif isa(Lambda,Array) || isa(Lambda,NTuple)
        if length(Lambda)~=d[4]
            error("incompatible size of lambdal")
        end
        lambdal = (Float32.(lambdal)..., zeros(Float32,nlam-length(lambdal))...)
    elseif numel(lambdal)==1
        error("wrong type of lambdal")
    end

    d1      = UInt64.(ceil.(d[1:3].-2)./2)
    threads, blocks = getthreads(d1, cutv3d)

    setindex!(CuGlobal{NTuple{   4, UInt64}}(tvmod,"d"),        UInt64.(d))
    setindex!(CuGlobal{NTuple{   3, Float32}}(tvmod,"vox"),     Float32.(vox))
    setindex!(CuGlobal{NTuple{nlam, Float32}}(tvmod,"lambdap"), Float32.(lambdap))
    setindex!(CuGlobal{NTuple{nlam, Float32}}(tvmod,"lambdal"), Float32.(lambdal))
    gl_o = CuGlobal{NTuple{3,UInt64}}(tvmod,"o")
    for it=1:nit
        for ok=0:2
            for oj=0:2
                for oi=0:2
                    setindex!(gl_o, UInt64.((oi,oj,ok)))
                    cudacall(cutv3d, (CuPtr{Cfloat},CuPtr{Cfloat}),
                             pointer(y), pointer(x);
                             threads=threads, blocks=blocks)
                end
            end
        end
    end
    return y
end
#==========================================================================

==========================================================================#
function getthreads(d::CuDim, fun::CuFunction, shmem::Integer=0, dev::CuDevice=CUDA.device())
    # Additional complications because this code uses 3D threads and blocks
    # rather than the much simpler 1D implementation

    config = launch_configuration(fun; shmem=shmem, max_threads=prod(d))
    nmax   = config.threads

    if isa(d,Integer)
        threads = nmax
        blocks  = Int32.(ceil.(d./threads))
    else
        @assert(length(d)==3,"Wrong sized dimensions.")
        s = ones(Int64,length(d))
        c = cumprod(d)
        if c[1]>nmax
            s[1] = min(nmax, attribute(dev,CUDA.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X))
        else
            s[1] = d[1]
            if c[2]>nmax
                s[2] = min(floor(nmax/c[1]), attribute(dev,CUDA.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y))
            else
                s[2] = d[2]
                if c[3]>nmax
                    s[3] = min(floor(Int64,nmax/c[2]), attribute(dev,CUDA.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z))
                else
                    s[3] = d[3]
                end
            end
        end
        threads = (s...,)
        blocks  = Int32.(ceil.(d./threads))
    end
    return threads, blocks
end

