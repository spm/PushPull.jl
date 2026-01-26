
function TVdenoise(x::Union{Array{Float32,3},Array{Float32,4}}, nit::Integer=1, vox::NTuple{3,Real}=(1.0f0,1.0f0,1.0f0), lambda::Union{Real,Array{Real}}=1.0f0)
    y = deepcopy(x)
    y = TVdenoise!(x,y,nit,vox,lambda)
    return y
end

function TVdenoise!(x::Union{Array{Float32,3},Array{Float32,4}}, y::Union{Array{Float32,3},Array{Float32,4}}, nit::Integer, vox::NTuple{3,Real}=(1.0f0,1.0f0,1.0f0), lambdap::Union{Real,Array{Real}}=1.0f0, lambdal::Union{Real,Array{Real}}=1.0f0)

    global tvlib
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

    for it=1:nit
        for ok=0:2
            for oj=0:2
                for oi=0:2
                    GC.@preserve y x d vox lambdap lambdal  begin
                        ccall(dlsym(tvlib,:TVdenoise3d), Cvoid,
                              (Ref{Cfloat}, Ptr{Cfloat}, Ptr{Csize_t}, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}),
                              pointer(y), pointer(x), pointer(d), pointer(vox), pointer(lambdap), pointer(lambdal))
                    end
                end
            end
        end
    end
    return y
end
#==========================================================================

==========================================================================#

