using CUDA
#using Libdl # Later

const KernelType = NamedTuple{(:stride, :d, :nchan, :offset, :length, :values, :indices, :patch_indices),
                              Tuple{NTuple{3, Int64}, NTuple{3, Int64}, Int64, Matrix{Int32}, Matrix{Int32},
                              Vector{Float32}, Vector{Int32}, Vector{Int32}}}

function setkernel(kernel::KernelType, bnd)
#=
    #define MAX_ELEM 256
    #define MAXN 8
    #define MAXD 5
    #define BUFLEN 125

    __constant__ float   values[MAX_ELEM];        /* Values in sparse matrix*/
    __constant__ int     indices[MAX_ELEM];       /* Indices into images */
    __constant__ int     patch_indices[MAX_ELEM]; /* Indices into patchs */
    __constant__ int     offset[MAXN*MAXN];       /* Offsets into values/indices */
    __constant__ int     length[MAXN*MAXN];       /* Number of non-zero off diagonal elements */
    __constant__ int     bnd[3*MAXN];             /* Boundary conditions */
    __constant__ USIZE_t  d[5];                   /* image data dimensions */
    __constant__ USIZE_t dp[3];                   /* filter dimensions */
    __constant__ USIZE_t  o[3];                   /* offsets into volume */
    __constant__ USIZE_t  n[3];                   /* number of elements */
=#
    global opmod

    maxn     = 8
    maxd     = 5
    max_elem = 256

    @assert(length(kernel.values)<=max_elem)

    r         = 1:length(kernel.values)
    values    = zeros(Float32, max_elem)
    values[r] = kernel.values[:]
    setindex!(CuGlobal{NTuple{max_elem,Float32}}(opmod,"values"), (values...,))
    indices    = zeros(Int32, max_elem)
    indices[r] = kernel.indices[:]
    setindex!(CuGlobal{NTuple{max_elem,Int32}}(opmod,"indices"), (indices...,))
    indices[r] = kernel.patch_indices[:]
    setindex!(CuGlobal{NTuple{max_elem,Int32}}(opmod,"patch_indices"), (indices...,))

    r      = 1:length(kernel.offset)
    offset = zeros(Int32,maxn*maxn)
    offset[r] = kernel.offset[:]
    setindex!(CuGlobal{NTuple{maxn*maxn,Int32}}(opmod,"offset"), (offset...,))
    len       = zeros(Int32,maxn*maxn)
    len[r]    = kernel.length[:]
    setindex!(CuGlobal{NTuple{maxn*maxn,Int32}}(opmod,"length"), (len...,))

    bndf = zeros(Int32,3*maxn)
    bndf[1:length(bnd)] = bnd[:]
    bndf = (bndf...,)
    setindex!(CuGlobal{NTuple{3*maxn,Int32}}(opmod,"bnd"),  bndf)
end


function vel2mom(v::AbstractArray{Float32,4},
                 kernel::KernelType,
                 bnd::Array{<:Integer} = Int32.([2 1 1; 1 2 1; 1 1 2]))

    u  = zero(v)
    vel2mom!(u, v, kernel, bnd)
    return u
end

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

    ccall(dlsym(oplib,:vel2mom), Cvoid,
          (Ref{Cfloat}, Ptr{Cfloat}, Ptr{Csize_t},
           Ptr{Cint}, Ptr{Cint},
           Ptr{Cfloat}, Ptr{Cint}, Ptr{Cint}, Ptr{Csize_t}, Ptr{Cint}),
          pointer(u), pointer(v), pointer(d),
          pointer(kernel.offset), pointer(kernel.length),
          pointer(kernel.values), pointer(kernel.indices), pointer(kernel.patch_indices), pointer(dp), pointer(bnd))
     return u
end

function vel2mom!(u::CuArray{Float32,4},
                  v::CuArray{Float32,4},
                  kernel::KernelType,
                  bnd::Array{<:Integer} = Int32.([2 1 1; 1 2 1; 1 1 2]))

    function run_kernel(fun, threads, r, u, v)
        # Computations using zero-offset
        o  = UInt64.(first.(r))
        n  = UInt64.(max.((last.(r) .- first.(r)),0))
        n1 = prod(n)
        if n1>0
            setindex!(CuGlobal{NTuple{3,UInt64}}(opmod,"o"), o)
            setindex!(CuGlobal{NTuple{3,UInt64}}(opmod,"n"), n)

            n1      = prod(n)
            threads = min(threads,n1)
            blocks  = Int32(ceil(n1./threads))
            cudacall(fun, (CuPtr{Cfloat},CuPtr{Cfloat}),
                     pointer(u), pointer(v);
                     threads=threads, blocks=blocks)
            #print("(",threads,"/",blocks,")")
        end
        return nothing
    end

    @assert(all(size(u)  .== size(v)))
    @assert(all(kernel.d .== size(v)[1:3]))
    global opmod
    global cuVel2mom
    global cuVel2momPad

    # Put some things in constant memory for speed
    setindex!(CuGlobal{NTuple{5,UInt64}}(opmod,"d"),  UInt64.((kernel.d...,3,3)))
    setindex!(CuGlobal{NTuple{3,UInt64}}(opmod,"dp"), UInt64.(kernel.stride))
    setkernel(kernel,bnd)

    threads_pad   = launch_configuration(cuVel2momPad).threads
    threads_nopad = launch_configuration(cuVel2mom).threads

    d  = kernel.d
    dp = kernel.stride
    rs = min.(Int.((dp.+1)./2),d)     # Start of middle block
    re = max.(rs,Int.(d.-(dp.+1)./2)) # End of middle block

    if any(re.<=rs)
        # No middle block
        r = UnitRange.(0,d)
        run_kernel(cuVel2momPad, threads_pad, r, u, v)
    else
        r  = (UnitRange(0,d[1]), UnitRange(0,d[2]), UnitRange(0,rs[3]))
        run_kernel(cuVel2momPad, threads_pad, r, u, v)

        r  = (UnitRange(0,d[1]), UnitRange(0,rs[2]), UnitRange(rs[3],re[3]))
        run_kernel(cuVel2momPad, threads_pad, r, u, v)

        r  = (UnitRange(0,rs[1]), UnitRange(rs[2],re[2]), UnitRange(rs[3],re[3]))
        run_kernel(cuVel2momPad, threads_pad, r, u, v)

        # Middle block
        r  = UnitRange.(rs,re)
        run_kernel(cuVel2mom, threads_nopad, r, u, v)

        r  = (UnitRange(re[1],d[1]), UnitRange(rs[2],re[2]), UnitRange(rs[3],re[3]))
        run_kernel(cuVel2momPad, threads_pad, r, u, v)

        r  = (UnitRange(0,d[1]), UnitRange(re[2],d[2]), UnitRange(rs[3],re[3]))
        run_kernel(cuVel2momPad, threads_pad, r, u, v)

        r  = (UnitRange(0,d[1]), UnitRange(0,d[2]), UnitRange(re[3],d[3]))
        run_kernel(cuVel2momPad, threads_pad, r, u, v)
    end
    return v
end


function relax!(g::Array{Float32,4}, h::Array{Float32,4}, kernel::KernelType,
                bnd::Array{<:Integer}, nit::Integer, v::Array{Float32,4}) 
    global oplib
    (d, dp) = checkdims(g,h,v,kernel,bnd)
    bnd = Int32.(bnd[:])
    for it=1:nit
        ccall(dlsym(oplib,:relax), Cvoid,
              (Ref{Cfloat}, Ptr{Csize_t}, Ptr{Cfloat}, Ptr{Cfloat},
               Ptr{Cint}, Ptr{Cint}, Ptr{Cfloat},
               Ptr{Cint}, Ptr{Cint}, Ptr{Csize_t}, Ptr{Cint}),
              pointer(v), pointer(d), pointer(g), pointer(h), 
              pointer(kernel.offset), pointer(kernel.length), pointer(kernel.values),
              pointer(kernel.indices), pointer(kernel.patch_indices), pointer(dp), pointer(bnd))
    end
    return v
end


function relax!(g::CuArray{Float32,4}, h::CuArray{Float32,4}, kernel::KernelType,
                bnd::Array{<:Integer}, nit::Integer, v::CuArray{Float32,4})

    global opmod
    global cuRelax
    global cuRelaxPad

    (d, dp) = checkdims(g,h,v,kernel,bnd)

    # Put some things in constant memory for speed
    setindex!(CuGlobal{NTuple{5,UInt64}}(opmod,"d"),  d)
    setindex!(CuGlobal{NTuple{3,UInt64}}(opmod,"dp"), dp)
    setkernel(kernel,bnd)

    regions  = range.(d[1:3],dp[1:3])

    config        = launch_configuration(cuRelaxPad)
    threads_pad   = config.threads
    config        = launch_configuration(cuRelax)
    threads_nopad = config.threads

    for it=1:nit
        for k=1:length(regions[3])
            rk = regions[3][k]
            for j=1:length(regions[2])
                rj = regions[2][j]
                for i=1:length(regions[1])
                    ri = regions[1][i]
                    r  = (start=(ri[1],rj[1],rk[1]), step=dp, stop=(ri[2],rj[2],rk[2]))
                    fun,threads = (k==2 && j==2 && i==2) ? (cuRelax,threads_nopad) : (cuRelaxPad,threads_pad)
                    relax_block!(v, g, h, fun, r, threads)
                end
            end
        end
    end
    return v
end


function relax_block!(v::CuArray{Float32,4},
                      g::CuArray{Float32,4},
                      h::CuArray{Float32,4},
                      fun::CuFunction,
                      r, threads0=0)
    global opmod

    if threads0==0
        config   = launch_configuration(fun)
        threads0 = config.threads
    end

    for k=1:min(r.step[3],r.stop[3]-r.start[3]+1),
        j=1:min(r.step[2],r.stop[2]-r.start[2]+1),
        i=1:min(r.step[1],r.stop[1]-r.start[1]+1)
        o  = UInt64.(r.start .+ (i,j,k) .- 2)
        n  = UInt64.(floor.((r.stop.-(o.+1))./r.step.+1))
        n1 = prod(n)
        threads = min(threads0,n1)
        blocks  = Int32(ceil(n1./threads))
        setindex!(CuGlobal{NTuple{3,UInt64}}(opmod,"o"), o)
        setindex!(CuGlobal{NTuple{3,UInt64}}(opmod,"n"), n)
        cudacall(fun, (CuPtr{Cfloat},CuPtr{Cfloat},CuPtr{Cfloat}),
                 pointer(v), pointer(g), pointer(h);
                 threads=threads, blocks=blocks)
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


