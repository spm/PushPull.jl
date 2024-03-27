using FFTW

"""

    L = registration_operator(vx::Vector{<:Real}, λ::Vector{<:Real})

Generate differential operator for image registration.

Requirements
* `vx` must contain three elements that denote voxel sizes in mm
* `λ` must conatain four elements, denoting
    - `λ[1]` Penalty on displacements/velocities (positive)
    - `λ[2]` Penalty on "membrane energy" of displacements/velocities (positive)
    - `λ[3]` Penalty on "bending energy" of displacements/velocities (positive)
    - `λ[4]` Penalty on divergences of displacements/velocities (positive or negative)

If `λ[4]` is nonzero, the output dimensions of `L` are 5×5×5×D, otherwise they are 3×3×3×D.
If `λ[3]` is nonzero, the above D is 6, otherwise it is 3 (enough to encode the
unique elements of a 3×3 symmetric matrix, rather than just the diagonals of that matrix).

# Membrane energy
This regularisation (technically known as the Dirichlet energy) is widely used for deep learning-based image registration.

E = ∫ₓ||∇𝐯(𝐱)||² d𝐱/2

This can be achieved by having all elements of `λ` set to zero, except for `λ[2]`.

# Bending energy
This form of regularisation can be achieved by having all elements of `λ` set to zero, except for `λ[3]`.
In the following, Δ = ∇² = ∇.∇

E = ∫ₓ||Δ𝐯(𝐱)||² d𝐱/2

# Beg's regulariser
In the Large Deformation Metric Mapping paper, regularisation was of the form

E = ∫ₓ||(-αΔ + γ)𝐯(𝐱)||² d𝐱/2

This can be achieved with
* `λ[1]` = γ²
* `λ[2]` = 2αγ
* `λ[3]` = α²

# Linear Elasticity
For linear elasticity, use `λ[2]` and `λ[4]`, which encode the Lamé constants.
This is often expressed along the lines of

E = ∫ₓ 𝐯(𝐱)ᵀ(μ∇² + (μ+λ)∇∇ᵀ)𝐯(𝐱) d𝐱/2

With our definitions, this would be

E = ∫ₓ 𝐯(𝐱)ᵀ (`λ[2]` ∇² + `λ[4]` ∇∇ᵀ)𝐯(𝐱) d𝐱/2

Linear elastic regularisation can be achieved with
* Shear modulus (usually denoted by μ) = `λ[2]`
* Lamé's first parameter (often denoted by λ) = `λ[2]`+`λ[4]`

"""
function registration_operator(vx::Vector{<:Real}, λ::Vector{<:Real})
    @assert(length(vx)==3,"Wrong sized vx.")
    @assert(length(λ)==4, "Wrong sized λ.")
    d123 = any(λ[[2,4]].!=0) ? 3 : 1
    d123 = λ[3]!=0 ? 5 : d123
    d4   = λ[4]==0 ? 3 : 6
    L    = zeros(Float32,(d123,d123,d123,d4))
    c    = Int((size(L,1)+1)/2)
    Δ  = [-1, 2, -1] # Laplacian ∇²
    r3 = (c-1):(c+1)

    if λ[1] != 0
        r = c:c
        for i=1:3
            L[c,c,c,i] += λ[1]
        end
    end
    if λ[2] != 0
        # Membrane energy
        # λ[2] corresponds with the shear modulus
        r3 = (c-1):(c+1)
        for i=1:3
            L[r3,c,c,i] .+= Δ*(λ[2]*(vx[i]/vx[1])^2) # ∂²vᵢ/∂x₁²
            L[c,r3,c,i] .+= Δ*(λ[2]*(vx[i]/vx[2])^2) # ∂²vᵢ/∂x₂²
            L[c,c,r3,i] .+= Δ*(λ[2]*(vx[i]/vx[3])^2) # ∂²vᵢ/∂x₃²
        end
    end
    if λ[3] != 0
        # Bending energy
        r5  = (c-2):(c+2)
        ΔΔ  = [1,-4,6,-4,1] # Δ * Δ
        ΔΔᵀ = Δ*Δ'          # Δ * Δᵀ
        for i=1:3
            L[r5, c, c,i] .+= ΔΔ*(λ[3]*(vx[i]/vx[1]^2)^2)
            L[ c,r5, c,i] .+= ΔΔ*(λ[3]*(vx[i]/vx[2]^2)^2)
            L[ c, c,r5,i] .+= ΔΔ*(λ[3]*(vx[i]/vx[3]^2)^2)
            L[r3,r3, c,i] .+= ΔΔᵀ*(2*λ[3]*(vx[i]/(vx[1]*vx[2]))^2)
            L[r3, c,r3,i] .+= ΔΔᵀ*(2*λ[3]*(vx[i]/(vx[1]*vx[3]))^2)
            L[ c,r3,r3,i] .+= ΔΔᵀ*(2*λ[3]*(vx[i]/(vx[2]*vx[3]))^2)
        end
    end
    if λ[4] != 0
        # Squared divergence
        # λ[4]-λ[2] corresponds with Lame's first parameter
        ∇   = [-1/2, 0, 1/2]  # Gradient operator ∂/∂x
        ∇∇ᵀ = ∇*∇'
        L[r3, c, c,1] .+= λ[4]*Δ   # ∂²v₁/∂x₁²
        L[ c,r3, c,2] .+= λ[4]*Δ   # ∂²v₂/∂x₂²
        L[ c, c,r3,3] .+= λ[4]*Δ   # ∂²v₃/∂x₃²
        L[r3,r3, c,4] .+= λ[4]*∇∇ᵀ # ∂²v₃/∂x₁∂x₂ ??
        L[r3, c,r3,5] .+= λ[4]*∇∇ᵀ # ∂²v₂/∂x₁∂x₃ ??
        L[ c,r3,r3,6] .+= λ[4]*∇∇ᵀ # ∂²v₁/∂x₂∂x₃ ??
    end
    L
end
"""
    reduce2fit!(L::Array{<:Real,4}, d::NTuple{3,Integer})

Take the regularisation operator `L`, and (if necessary) reduce its
dimensions so that it fits with a set of image dimensions `d`.

"""
function reduce2fit!(L::Union{Array{<:Real,4},CuArray{<:Real,4}}, d::NTuple{3,Integer},
                     bnd::Array{<:Integer} = Int32.([2 1 1; 1 2 1; 1 1 2]))
    dp = size(L)
    c  = Int32.((dp[1:3].+1)./2)
    r0 = max.(c.-d,0)

    for dim=1:3
        if bnd[1,dim]==0 || d[1]==1
            r  = [1:r0[1]; (dp[1]+1-r0[1]):dp[1]]
        else
            r = []
        end
        if ~isempty(r)
            if bnd[1,dim]==2
                w                   = ones(dp[1])
                w[[c[1]-1,c[1]+1]] .= -1
                w = reshape(w[r],(length(r),1,1))
                L[[c[1]],:,:,dim] .+= sum(L[r,:,:,dim].*w,dims=1)
            else
                L[[c[1]],:,:,dim] .+= sum(L[r,:,:,dim],dims=1)
            end
            L[r,:,:,dim]  .= 0
        end

        if bnd[2,dim]==0 || d[2]==1
            r  = [1:r0[2]; (dp[2]+1-r0[2]):dp[2]]
        else
            r = []
        end
        if ~isempty(r)
            if bnd[2,dim]==2
                w                   = ones(dp[2])
                w[[c[2]-1,c[2]+1]] .= -1
                w = reshape(w[r],(1,length(r),1))
                L[:,[c[2]],:,dim] .+= sum(L[:,r,:,dim].*w,dims=2)
            else
                L[:,[c[2]],:,dim] .+= sum(L[:,r,:,dim],dims=2)
            end
            L[:,r,:,dim]  .= 0
        end

        if bnd[3,dim]==0 || d[3]==1
            r  = [1:r0[3]; (dp[3]+1-r0[3]):dp[3]]
        else
            r  = []
        end
        if ~isempty(r)
            if bnd[3,dim]==2
                w                   = ones(dp[3])
                w[[c[3]-1,c[3]+1]] .= -1
                w = reshape(w[r],(1,1,length(r)))
                L[:,:,[c[3]],dim] .+= sum(L[:,:,r,dim].*w,dims=3)
            else
                L[:,:,[c[3]],dim] .+= sum(L[:,:,r,dim],dims=3)
            end
            L[:,:, r,dim]  .= 0
        end
    end
    msk = sum(L.!=0,dims=4)
    i1  = sum(msk,dims=(2,3))[:] .!= 0
    i2  = sum(msk,dims=(1,3))[:] .!= 0
    i3  = sum(msk,dims=(1,2))[:] .!= 0
    return L[i1[:],i2[:],i3[:],:]
end


"""
    sparsify(L::Array{<:Real,4}, d::NTuple{3,Integer}, nd=3)

Take a dense convolution operator `L` and convert it into a sparse representation
suitable for doing faster convolution/relaxation.

Returns a named tuple:

    NamedTuple{(:stride,               :d,              :nchan, :offset,      :length,       :values,         :indices,      :patch_indices),
                Tuple{NTuple{3,Int64}, NTuple{3,Int64}, Int64,  Matrix{Int64}, Matrix{Int64}, Vector{Float32}, Vector{Int32}, Vector{Int32}}}
where:

* `:stride`:  Dimensions of the filter kernel.
* `:d`:       Image dimensions.
* `:nchan`:   Number of channels (3).
* `:offset`:  `:nchan` × `:nchan` matrix of offsets.
* `:length`:  `:nchan` × `:nchan` matrix of lengths.
* `:values`:  Values from the filter.

    For each input `i` and output `j` dimension, the elements of the values are given by:

    `values[offset[i,j].+(1:length[i,j])]`

    The values at the centre of the filters are always stored, even if they have a value
    of zero. This means that length[i,j] ≥ 1.

    Note that the sums of the values are also stored because they are used for a more
    stable relaxation procedure:

    `values[offset[i,j]] = sum(values[offset[i,j].+(1:length[i,j])])`

* `:indices`: Indices into image sized arrays.

    Indices, relative to the centre voxel, may be negative and are given by:

    `indices[offset[i,j].+(1:length[i,j])]`

* `:patch_indices`: Indices into patch-sized arrays.

    Indices within a patch of size `:stride` are numbered from 0 (because of CUDA code), and given by

    `patch_indices[offset[i,j].+(1:length[i,j])]`

"""
function sparsify(L::Array{<:Real,4}, d::NTuple{3,Integer}, nd=3)
    dp = (size(L,1),size(L,2),size(L,3))
    @assert(all(rem.(dp,2).==1),"First three dimensions of `L` must be odd")
    @assert(size(L,4)==3 || size(L,4)==6, "Incorrectly sized `L`")

    function rearrange(nnz::Array{<:Integer,2}, A::Array{T,3}) where T
        o   = zero(nnz)
        B   = zeros(T,sum(nnz)+length(nnz))
        to  = 1
        for j=1:size(nnz,2), i=1:size(nnz,1)
            if isa(B,Array{<:AbstractFloat})
                B[to]  = sum(A[2:nnz[i,j],i,j])
            end
            B[(to+1):(to+nnz[i,j])] = A[1:nnz[i,j],i,j];
            o[i,j] = to;
            to += nnz[i,j]+1
        end
        return B, o
    end

    image_offset(i,j,k,c1) = i-cv[1] +  d[1]*(j-cv[2] +  d[2]*(k-cv[3] +  d[3]*(c1-1)))
    patch_offset(i,j,k)    = i-1     + dp[1]*(j-1     + dp[2]*(k-1))

    maxnnz = size(L,1)*size(L,2)*size(L,3)
    Ov  = zeros(Float32, maxnnz, nd,nd)
    Oi  = zeros(Int32,   maxnnz, nd,nd)
    Pi  = zeros(Int32,   maxnnz, nd,nd)

    nnz = zeros(Int32,  nd,nd)
    cv  = Int32.((dp.+1)./2)

    for c=1:nd
        nnz[c,c]        += 1
        Ov[nnz[c,c],c,c] = L[cv[1],cv[2],cv[3],c]
        Oi[nnz[c,c],c,c] = image_offset(cv[1],cv[2],cv[3],c)
        Pi[nnz[c,c],c,c] = patch_offset(cv[1],cv[2],cv[3])

        for k=1:size(L,3), j=1:size(L,2), i=1:size(L,1)
            if i!=cv[1] || j!=cv[2] || k!=cv[3]
                if L[i,j,k,c] != 0
                    nnz[c,c]        += 1
                    Ov[nnz[c,c],c,c] = L[i,j,k,c]
                    Oi[nnz[c,c],c,c] = image_offset(i,j,k,c)
                    Pi[nnz[c,c],c,c] = patch_offset(i,j,k)
                end
            end
        end
    end

    if size(L,4)==Int(nd*(nd+1))/2
        c = nd
        for c1=1:nd, c2=(c1+1):nd
            c          += 1
            nnz[c1,c2] += 1
            nnz[c2,c1] += 1
            Ov[nnz[c1,c2],c1,c2] = Ov[nnz[c2,c1],c2,c1] = L[cv[1],cv[2],cv[3],c]
            Oi[nnz[c1,c2],c1,c2] = image_offset(cv[1],cv[2],cv[3],c1)
            Oi[nnz[c2,c1],c2,c1] = image_offset(cv[1],cv[2],cv[3],c2)
            Pi[nnz[c1,c2],c1,c2] = Pi[nnz[c2,c1],c2,c1] = patch_offset(cv[1],cv[2],cv[3])

            for k=1:size(L,3), j=1:size(L,2), i=1:size(L,1)
                if i!=cv[1] || j!=cv[2] || k!=cv[3]
                    if L[i,j,k,c] != 0
                        nnz[c1,c2] += 1
                        nnz[c2,c1] += 1
                        Ov[nnz[c1,c2],c1,c2] = Ov[nnz[c2,c1],c2,c1] = L[i,j,k,c]
                        Oi[nnz[c1,c2],c1,c2] = image_offset(i,j,k,c1)
                        Oi[nnz[c2,c1],c2,c1] = image_offset(i,j,k,c2)
                        Pi[nnz[c1,c2],c1,c2] = Pi[nnz[c2,c1],c2,c1] = patch_offset(i,j,k)
                    end
                end
            end
        end
    else
        for c1=1:nd, c2=(c1+1):nd
            nnz[c1,c2]          += 1
            nnz[c2,c1]          += 1
            Ov[nnz[c1,c2],c1,c2] = Ov[nnz[c2,c1],c2,c1] = 0.0
            Oi[nnz[c1,c2],c1,c2] = Oi[nnz[c2,c1],c2,c1] = image_offset(cv[1],cv[2],cv[3],c2)
            Pi[nnz[c1,c2],c1,c2] = Pi[nnz[c2,c1],c2,c1] = patch_offset(cv[1],cv[2],cv[3])
        end
    end
    Ov,  = rearrange(nnz,Ov)
    Oi,o = rearrange(nnz,Oi)
    Pi,o = rearrange(nnz,Pi)
    kernel = (stride=(size(L,1),size(L,2),size(L,3)), d=d, nchan=nd, offset=o, length=nnz, values=Ov, indices=Oi, patch_indices=Pi)
    return kernel
end

"""
    padft(L::Array{Float32,3},d::NTuple{3,Integer}, r::NTuple{3,UnitRange}=UnitRange.(1,d.+1))

3D padded FFT that returns the fft of the operator padded out to size `2.*d`.
The elements that are returned are defined by `r`.

"""

#using Dates
#tim = ()->Dates.now().instant.periods.value

function free!(x::AbstractArray)
    if isa(x,CuArray)
        CUDA.unsafe_free!(x)
    end
    return nothing
end


"""
"""
function greens(L::Union{CuArray{Float32,4},Array{Float32,4}}, d::NTuple{3,Integer})

    function scratchlen(dl,d)
        # Could re-order the fft to reduce memory requirements
        d0  = [dl...]
        d1  = deepcopy(d0)
        len1 = 0
        for i=1:3
            d1[i] = 2*d[i]
            len1  = max(len1,prod(d1))
            d1[i] = d[i]+1
        end
        len0 = 0
        for i=1:3
            d0[i] = d[i]+1
            len0  = max(len0,prod(d0))
        end
        return len0,len1
    end

    dl = size(L)
    # TODO: Determine the actual maximum amount of memory required
    #sl = 2*prod(d.+1)+prod(d.+1)
    sl  = sum(scratchlen(size(L)[1:3],d))
    if isa(L,CuArray)
        K       = Array{CuArray{Float32,3}}(undef, dl[4])
        scratch = CUDA.zeros(ComplexF32,sl)
    else
        K       = Array{  Array{Float32,3}}(undef, dl[4])
        scratch = zeros(ComplexF32,sl)
    end

    function padft(L::Union{AbstractArray{Float32,3},AbstractArray{Complex{Float32},3}},
                   d::NTuple{3,Integer}, r::NTuple{3,UnitRange}=UnitRange.(1,d.+1))

        function scratch_array(dl,d,r,dim)
            dl0      = [dl...]
            dl0[dim] = length(r[dim])
            L0       = reshape(view(scratch,1:prod(dl0)),dl0...)
            o,unused = scratchlen(dl,d)
            dl0[dim] = 2*d[dim]
            L1       = reshape(view(scratch,(o+1):(o+prod(dl0))),dl0...)
            L1      .= 0
            return L0, L1
        end

        function padft_dim(L,d,dim,r)
            dl1      = [size(L)...]
            dl1[dim] = 2*d[dim]
            T        = eltype(L)
            T        = T<:Complex ? T : Complex{T}
           #L1       = isa(L,CuArray) ? CUDA.zeros(T, (dl1...)) : zeros(T, (dl1...))
            L0,L1    = scratch_array(size(L),d,r,dim)
            bc       = 0
            c        = round(Int,(size(L,dim)+1)/2)
            indices  = mod.((1:size(L,dim)).-c, 2*d[dim]).+1
            ind0     = [UnitRange.(1,size(L))...]
            ind1     = deepcopy(ind0)
            for i=1:length(indices)
                ind1[dim]          = indices[i]:indices[i]
                ind0[dim]          = i:i
                view(L1,ind1...) .+= view(L,ind0...)
            end
            fft!(L1,dim)
            ind0[dim] = r[dim]
            L0       .= view(L1, ind0...)
            return L0
        end

        for dim=1:3
            L = padft_dim(L,d,dim,r)
        end
        return real(L)
    end

    if dl[4]==3
        r   = [UnitRange.(1,d[1:3])...]
        for i=1:dl[4]
            ri     = r[i]
            r[i]   = 2:(d[i]+1)
            K[i]   = padft(view(L, :,:,:,i), d, (r...,))
            K[i] .^= (-1)
            r[i]   = ri
            if ~all(isfinite.(Array(view(K[i],1:1))))
                CUDA.@allowscalar K[i][1:1] .= 0.0f0
            end
        end
    else
        if isa(L,CuArray)
            F = Array{CuArray{Float32,3}}(undef, dl[4])
        else
            F = Array{  Array{Float32,3}}(undef, dl[4])
        end
        r0   = UnitRange.(1,d[1:3].+1)
        for i=1:dl[4]
            F[i] = padft(view(L, :,:,:,i), d, r0)
        end

        # Re-use scratch
        # Note that dF and t are real, but scratch is complex.
        # Need to figure out a way of using the same memory as either real or complex.
        d1   = d[1:3].+1
        dF   = reshape(view(scratch, 1:prod(d1)), d1...)
        t    = reshape(view(scratch, (length(dF)+1):(2*length(dF))), size(dF)...)

        # Determinants
        #dF .= 1 ./ (F[1].*F[2].*F[3] .+ 2 .*F[4].*F[5].*F[6] .- F[3].*F[4].^2 .- F[2].*F[5].^2 .- F[1].*F[6].^2)
        dF  .= F[1].*(F[2].*F[3] .- F[6].^2) .+ F[4].*(2 .*F[5].*F[6] .- F[3].*F[4]) .- F[2].*F[5].^2
        dF .^= -1
        if ~all(isfinite.(Array(view(dF,1:1))))
            CUDA.@allowscalar dF[1:1] .= 0.0f0
        end

        # "diagonal" components
        t    .= (F[2].*F[3] .- F[6].^2).*dF
        tv    = view(t, 2:d[1]+1, 1:d[2], 1:d[3])
        K[1]  = typeof(F[1])(undef,size(tv))
        K[1] .= tv

        t    .= (F[1].*F[3] .- F[5].^2).*dF
        tv    = view(t, 1:d[1], 2:d[2]+1, 1:d[3])
        K[2]  = typeof(F[1])(undef,size(tv))
        K[2] .= tv

        t    .= (F[1].*F[2] .- F[4].^2).*dF
        tv    = view(t, 1:d[1], 1:d[2], 2:d[3]+1)
        K[3]  = typeof(F[1])(undef,size(tv))
        K[3] .= tv


        # "off-diagonal" components
        tv    = view(t, UnitRange.(1,d)...)

        t    .= (F[5].*F[6] .- F[3].*F[4]).*dF
        K[4]  = typeof(F[1])(undef,size(tv))
        K[4] .= tv

        t    .= (F[4].*F[6] .- F[2].*F[5]).*dF
        K[5]  = typeof(F[1])(undef,size(tv))
        K[5] .= tv

        t    .= (F[4].*F[5] .- F[1].*F[6]).*dF
        K[6]  = typeof(F[1])(undef,size(tv))
        K[6] .= tv
    end
    return K
end


function kernel(d::NTuple{3,Integer}, vx::Vector{<:Real}=[1,1,1], λ::Vector{<:Real}=[0,1,0,0])
    L = registration_operator(vx,λ)
    K = greens(L, d)
end

function mom2vel(u::AbstractArray{Float32,4}, K::Vector{<:AbstractArray{Float32,3}})
    v  = zero(u)                              # Memory allocation (3 × volumes)
    cl = dst_scratch(size(u)[1:3], typeof(u)) # Memory allocation (4 × volumes)
    if length(K)==3
        t           =  dct!( dst!(u[:,:,:,1], 1), (2,3))
        v[:,:,:,1] .= idct!(idst!(t .*= K[1], 1), (2,3))
        t          .=  dct!( dst!(u[:,:,:,2], 2), (1,3))
        v[:,:,:,2] .= idct!(idst!(t .*= K[2], 2), (1,3))
        t          .=  dct!( dst!(u[:,:,:,3], 3), (1,2))
        v[:,:,:,3] .= idct!(idst!(t .*= K[3], 3), (1,2))
    else
        U    = isa(u,CuArray) ? Array{CuArray{Float32,3}}(undef, 3) : Array{  Array{Float32,3}}(undef, 3)
        U[1] = dct!( dst!(u[:,:,:,1],1), (2,3)) # Memory allocation (1 × volume)
        U[2] = dct!( dst!(u[:,:,:,2],2), (1,3)) # Memory allocation (1 × volume)
        U[3] = dct!( dst!(u[:,:,:,3],3), (1,2)) # Memory allocation (1 × volume)

        d           = size(u)
        t           = K[1].*U[1] # Memory allocation (1 × volume)
        view(t, 1:d[1]-1,2:d[2],:) .+= view(K[4], 2:d[1], 2:d[2], 1:d[3]).*view(U[2], 2:d[1], 1:d[2]-1, 1:d[3])
        view(t, 1:d[1]-1,:,2:d[3]) .+= view(K[5], 2:d[1], 1:d[2], 2:d[3]).*view(U[3], 2:d[1], 1:d[2],   1:d[3]-1)
        v[:,:,:,1] .= idct!(idst!(t, 1), (2,3))

        t          .= K[2].*U[2]
        view(t, 2:d[1],1:d[2]-1,:) .+= view(K[4], 2:d[1], 2:d[2], 1:d[3]).*view(U[1], 1:d[1]-1, 2:d[2], 1:d[3])
        view(t, :,1:d[2]-1,2:d[3]) .+= view(K[6], 1:d[1], 2:d[2], 2:d[3]).*view(U[3], 1:d[1],   2:d[2], 1:d[3]-1)
        v[:,:,:,2] .= idct!(idst!(t, 2), (1,3))

        t          .= K[3].*U[3]
        view(t, 2:d[1],:,1:d[3]-1) .+= view(K[5], 2:d[1], 1:d[2], 2:d[3]).*view(U[1], 1:d[1]-1, 1:d[2],   2:d[3])
        view(t, :,2:d[2],1:d[3]-1) .+= view(K[6], 1:d[1], 2:d[2], 2:d[3]).*view(U[2], 1:d[1],   1:d[2]-1, 2:d[3])
        v[:,:,:,3] .= idct!(idst!(t, 3), (1,2))

    end
    dst_scratch(cl) # Free up scratch
    return v
end

