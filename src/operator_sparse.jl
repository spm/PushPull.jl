
const KernelType = NamedTuple{(:stride, :d, :nchan, :offset, :length, :values, :indices, :patch_indices),
                              Tuple{NTuple{3, Int64}, NTuple{3, Int64}, Int64, Matrix{Int32}, Matrix{Int32},
                              Vector{Float32}, Vector{Int32}, Vector{Int32}}}

function vel2mom(v::AbstractArray{Float32,4},
                 kernel::KernelType,
                 bnd::Array{<:Integer} = Int32.([2 1 1; 1 2 1; 1 1 2]))
    u  = zero(v)
    vel2mom!(u, v, kernel, bnd)
    return u
end

using Libdl

function vel2mom!(u::Array{Float32,4},
                  v::Array{Float32,4},
                  kernel::KernelType,
                  bnd::Array{<:Integer} = Int32.([2 1 1; 1 2 1; 1 1 2]))

    global oplib

    @assert(all(size(u).==size(v)))
    @assert(all(kernel.d .== size(v)[1:3]))
    @assert(kernel.nchan == size(v,4))
    d   = Csize_t.([size(v)..., 1])
    dp  = Csize_t.([kernel.stride...])
    bnd = Int32.(bnd[:])

    GC.@preserve u v d kernel dp bnd begin
        ccall(dlsym(oplib,:vel2mom), Cvoid,
              (Ref{Cfloat}, Ptr{Cfloat}, Ptr{Csize_t},
               Ptr{Cint}, Ptr{Cint},
               Ptr{Cfloat}, Ptr{Cint}, Ptr{Cint}, Ptr{Csize_t}, Ptr{Cint}),
              pointer(u), pointer(v), pointer(d),
              pointer(kernel.offset), pointer(kernel.length),
              pointer(kernel.values), pointer(kernel.indices), pointer(kernel.patch_indices), pointer(dp), pointer(bnd))
    end
    return u
end


function relax!(g::Array{Float32,4}, h::Array{Float32,4}, kernel::KernelType,
                bnd::Array{<:Integer}, nit::Integer, v::Array{Float32,4}) 
    global oplib
    (d, dp) = checkdims(g,h,v,kernel,bnd)
    bnd = Int32.(bnd[:])
    d   = [d...,]
    dp  = [dp...,]
    for it=1:nit
        GC.@preserve v d g h kernel dp bnd begin
            ccall(dlsym(oplib,:relax), Cvoid,
                  (Ref{Cfloat}, Ptr{Csize_t}, Ptr{Cfloat}, Ptr{Cfloat},
                   Ptr{Cint}, Ptr{Cint}, Ptr{Cfloat},
                   Ptr{Cint}, Ptr{Cint}, Ptr{Csize_t}, Ptr{Cint}),
                  pointer(v), pointer(d), pointer(g), pointer(h), 
                  pointer(kernel.offset), pointer(kernel.length), pointer(kernel.values),
                  pointer(kernel.indices), pointer(kernel.patch_indices), pointer(dp), pointer(bnd))
        end
    end
    return v
end


function checkdims(g::AbstractArray, h::AbstractArray, v::AbstractArray, kernel, bnd)
    maxd = 5

    if ndims(g) != ndims(h)
        (!isempty(v) && ndims(g) != ndims(v))
        error("incompatible dimensionality")
    end
    if ndims(g)>4 || ndims(h)>4 || (!isempty(v) && ndims(v)>4)
        error("too many dimensions")
    end
    if ndims(g)<3 || ndims(h)<3 || (!isempty(v) && ndims(v)<3)
        error("too few dimensions")
    end
    dg = size(g)
    dh = size(h)
    dv = size(v)

    if (size(bnd,1) != 3) || (size(bnd,2) != dg[4]) || !all(in.(bnd,((0,1,2),)))
        error("inappropriate boundary conditions")
    end

    if !isempty(h)
        if any(dh[1:3] .!= dg[1:3]) || (!isempty(v) && any(dv[1:3] != dg[1:3])) || any(kernel.d .!= dg[1:3])
            error("incompatible dimensions")
        end

        if size(h,4) == size(g,4)
            hcode = 1
        else
            hcode = 2
            if size(h,4) != (size(g,4)*(size(g,4)+1))/2
                error("incompatible hessian dimensions")
            end
        end
    else
        hcode = 0
    end

    if any(kernel.stride .> maxd)
        error("regulariser dimensions too big")
    end
    dp = UInt64.(kernel.stride)

    if ndims(g) == 4
        d = UInt64.((dg..., hcode))
    else
        d = UInt64.((dg..., 1, hcode))
    end

    return d, dp
end


