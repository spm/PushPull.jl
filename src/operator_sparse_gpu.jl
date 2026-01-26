using CUDA
global opmod

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
    cuVel2mom    = CuFunction(opmod, "_Z15vel2mom_elementPfPKf")
    cuVel2momPad = CuFunction(opmod, "_Z22vel2mom_padded_elementPfPKf")

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


function relax!(g::CuArray{Float32,4}, h::CuArray{Float32,4}, kernel::KernelType,
                bnd::Array{<:Integer}, nit::Integer, v::CuArray{Float32,4})

    function relax_block!(v::CuArray{Float32,4},
                          g::CuArray{Float32,4},
                          h::CuArray{Float32,4},
                          fun::CuFunction, r,
                          threads0=0)
        global opmod

        if threads0==0
            config   = launch_configuration(fun)
            threads0 = config.threads
        end

        #print(r[1][1],",",r[1][2],",",r[1][3]," -> ", r[3][1],",",r[3][2],",",r[3][3],"\n")
        for k=r[1][3]:(min(r[3][3],r[1][3]+r[2][3])-1),
            j=r[1][2]:(min(r[3][2],r[1][2]+r[2][2])-1),
            i=r[1][1]:(min(r[3][1],r[1][1]+r[2][1])-1)
            o  = (i,j,k)
            n  = Int64.(ceil.((Signed.(r[3]).-Signed.((i,j,k)))./Signed.(r[2])))
            if all(n.>0)
                n1      = prod(n)
                threads = min(threads0,n1)
                blocks  = Int32(ceil(n1./threads))
                setindex!(CuGlobal{NTuple{3,UInt64}}(opmod,"o"), UInt64.(o))
                setindex!(CuGlobal{NTuple{3,UInt64}}(opmod,"n"), UInt64.(n))
                cudacall(fun, (CuPtr{Cfloat},CuPtr{Cfloat},CuPtr{Cfloat}),
                         pointer(v), pointer(g), pointer(h);
                         threads=threads, blocks=blocks)
            end
        end
        return v
    end

    global opmod
    cuRelax      = CuFunction(opmod, "_Z13relax_elementPfPKfS1_")
    cuRelaxPad   = CuFunction(opmod, "_Z20relax_padded_elementPfPKfS1_")

    (d, dp) = checkdims(g,h,v,kernel,bnd)

    # Put some things in constant memory for speed
    setindex!(CuGlobal{NTuple{5,UInt64}}(opmod,"d"),  d)
    setindex!(CuGlobal{NTuple{3,UInt64}}(opmod,"dp"), dp)
    setkernel(kernel,bnd)

    threads_pad   = launch_configuration(cuRelaxPad).threads
    threads_nopad = launch_configuration(cuRelax).threads

    rs = min.(Int.((dp.-1)./2), Int64.(d[1:3]))     # Start of middle block
    re = max.(rs, Int.(Int64.(d[1:3]).-(dp.-1)./2)) # End of middle block

    block_range(is,ie,js,je,ks,ke) = (Signed.((is,js,ks)),Signed.(dp),Signed.((ie,je,ke)))
    #print("[",rs[1],",",rs[2],",",rs[3]," -> ",re[1],",",re[2],",",re[3],"]\n")

    for it=1:nit
        
        if any(re.<rs)
            # No middle block
            relax_block!(v,g,h,cuRelaxPad,block_range(    0, d[1],    0, d[2],    0, d[3]),threads_pad)
        else
            # Edge blocks
            relax_block!(v,g,h,cuRelaxPad,block_range(    0, d[1],    0, d[2],    0,rs[3]),threads_pad)
            relax_block!(v,g,h,cuRelaxPad,block_range(    0, d[1],    0,rs[2],rs[3],re[3]),threads_pad)
            relax_block!(v,g,h,cuRelaxPad,block_range(    0,rs[1],rs[2],re[2],rs[3],re[3]),threads_pad)
            relax_block!(v,g,h,cuRelaxPad,block_range(re[1], d[1],rs[2],re[2],rs[3],re[3]),threads_pad)
            relax_block!(v,g,h,cuRelaxPad,block_range(    0, d[1],re[2], d[2],rs[3],re[3]),threads_pad)
            relax_block!(v,g,h,cuRelaxPad,block_range(    0, d[1],    0, d[2],re[3], d[3]),threads_pad)

            # Middle block
            relax_block!(v,g,h,cuRelax,   block_range(rs[1],re[1],rs[2],re[2],rs[3],re[3]),threads_nopad)
        end
    end

    return v
end

