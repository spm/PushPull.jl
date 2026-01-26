import FFTW: r2r!, RODFT10, RODFT01, REDFT10, REDFT01, fft!, ifft!

# Attempt to use FFTW for Array
import FFTW
dct(x::Array{<:Number})     = FFTW.dct(x)
dct!(x::Array{<:Number})    = FFTW.dct!(x)
idct(x::Array{<:Number})    = FFTW.idct(x)
idct!(x::Array{<:Number})   = FFTW.idct!(x)
dct(x::Array{<:Number}, dims)   = FFTW.dct(x,dims)
dct!(x::Array{<:Number}, dims)  = FFTW.dct!(x,dims)
idct(x::Array{<:Number}, dims)  = FFTW.idct(x,dims)
idct!(x::Array{<:Number}, dims) = FFTW.idct!(x,dims)


function dct(X::AbstractArray{<:AbstractFloat}, dims=1:ndims(X))
    return dct!(deepcopy(X), dims)
end

function idct(X::AbstractArray{<:AbstractFloat}, dims=1:ndims(X))
    return idct!(deepcopy(X), dims)
end


scratch = []

#=
Note: FFTW.r2r! are not yet implemented for GPU, so code reverts to slower
      procedures when passed CuArrays. This requires additional memory
      in the form of a complex array that has twice the dimensions of the
      images.
TODO: Set up a complex scratch array for the CUDA dst/dct to work with
      to avoid additional memory allocations. Use a view of this memory
      for the faster (even dimension) dct.
=#


"""
    dst(X [, dims])

Performs a multidimensional type-II discrete sine transform (DCT) of the array X.
The optional dims argument specifies an iterable subset of dimensions (e.g. an
integer, range, tuple, or array) to transform along.
"""
function dst(X::AbstractArray{<:AbstractFloat}, dims=1:ndims(X))
    return dst!(deepcopy(X),dims)
end

"""
    idst(X [, dims])

Computes the multidimensional inverse discrete sine transform (DST) of the array X.
The optional dims argument specifies an iterable subset of dimensions (e.g. an
integer, range, tuple, or array) to transform along.
"""
function idst(X::AbstractArray{<:AbstractFloat}, dims=1:ndims(X))
    return idst!(deepcopy(X),dims)
end


"""
    dst!(X [, dims])
Same as dst, except that it operates in-place on X, which must be an array of real floating-point values.
"""
function dst!(X::Array{<:AbstractFloat},dims=1:ndims(X))
    X = _scale_post!(r2r!(X, RODFT10, dims), dims, 1)
end

function dst_scratch(d::NTuple{N, Integer}, dt::DataType) where {N}
    global _scratch_dst
    if ~@isdefined(_scratch_dst)
        _scratch_dst = []
    end
    constructor = Base.typename(dt).wrapper
    if dt==Array
        _scratch_dst = []
        return false
    else
        if (length(_scratch_dst) == 2*prod(d)) && (eltype(_scratch_dst)==Complex{eltype(dt)})
            # Already defined
            return false
        else
            #print("Allocating ", 2*prod(d)/1024, " elements of ", eltype(dt), ".\n")
            #_scratch_dst = CUDA.zeros(Complex{eltype(dt)}, 2*prod(d))
            _scratch_dst  = constructor{Complex{eltype(dt)}}(undef,2*prod(d))
            _scratch_dst .= 0
            return true
        end
    end
end

function dst_scratch(d::NTuple{N, Integer}, i::Integer) where {N}
    global _scratch_dst
    @assert(@isdefined(_scratch_dst))
    @assert(length(_scratch_dst) == 2*prod(d))
    d1    = [d...]
    d1[i] = 2*d[i]
    X2    = reshape(_scratch_dst, d1...)
    X1    = reshape(view(_scratch_dst,1:prod(d)),d)
    return X2, X1
end

function dst_scratch(do_clear::Bool=true)
    if do_clear
        global _scratch_dst = []
    end
    return nothing 
end

# Slow GPU implementation
function dst!(X::AbstractArray{<:AbstractFloat}, dims=1:ndims(X))
    dims = [dims...]
    d  = size(X);
    r1 = [StepRange.(1,1,d)...]
    cl = dst_scratch(d, typeof(X))
    for i in dims
        X2,   = dst_scratch(d,i)
        di    = d[i]
        ri    = r1[i]
        X2[r1...] .= view(X,r1...)
        r1[i] = (2*di):-1:(di+1)
        X2[r1...] .= .-X
        r1[i] = 2:1:(di+1)
        fft!(X2,i)
        w     = _dst_scale(X, d, i)
        X    .= imag.(view(X2,r1...).*w)
        r1[i] = ri
    end
    dst_scratch(cl)
    return X
end

"""
    idst!(X [, dims])

Same as idst, but operates in-place on X.
"""
function idst!(X::Array{<:AbstractFloat},dims=1:ndims(X))
    X = r2r!(_scale_pre!(X, dims, 1), RODFT01, dims)
end

# Slow GPU implementation
function idst!(X::AbstractArray{<:AbstractFloat}, dims=1:ndims(X))
    dims = [dims...]
    d  = size(X);
    r1 = [StepRange.(1,1,d)...]
    r2 = deepcopy(r1)
    cl = dst_scratch(d, typeof(X))
    for i in dims
        X2,   = dst_scratch(d,i)
        if true
            ri    = r1[i]
            w     = im ./ _dst_scale(X, d, i)
            r2[i] = r2[i].+1
            X2[r2...] .= X.*w
            w     = reshape(conj.(w[1:(end-1)]), _along(length(d), i, d[i]-1))
            r2[i] = (d[i]*2):-1:(d[i]+2)
            r1[i] = 1:1:(d[i]-1)
            X2[r2...] .= view(X,r1...).*w
            r2[i] = 1:1:1
            X2[r2...] .= 0
            r1[i] = ri
            ifft!(X2,i)
            X .= real.(view(X2,r1...))
            r2[i] = ri
        end
    end
    dst_scratch(cl)
    return X
end

# Slow GPU implementation
function dct!(X::AbstractArray{<:AbstractFloat}, dims=1:ndims(X))
    dims = [dims...]
    d  = size(X);
    r1 = [StepRange.(1,1,d)...]
    r2 = deepcopy(r1)

    cl = dst_scratch(d, typeof(X))

    for i in dims
        di = d[i]
        X2,X1 = dst_scratch(d, i)
        if di > 1
            if iseven(di)
                # Even sized
                r1i   = r1[i]
                r2i   = r2[i]
                r1[i] = 1:Int64(di/2)
                r2[i] = 1:2:di
                X1[r1...] .= view(X,r2...)
                r1[i] = Int64(di/2)+1:di
                r2[i] = di:-2:2
                X1[r1...] .= view(X,r2...)
                r1[i] = r1i
                r2[i] = r2i
                fft!(X1, i)
                w     = _dct_scale(X, d, i)
                w   .*= 2
                X   .= real.(X1.*w)
            else
                # Odd sized
                r1i   = r1[i]
                X2[r1...] .= X
                r1[i] = (2*di):-1:(di+1)
                X2[r1...] .= X
                r1[i] = r1i
                fft!(X2,i)
                w     = _dct_scale(X, d, i)
                X .= real.(view(X2,r1...).*w)
            end
        end
    end
    dst_scratch(cl)
    return X
end


# Slow GPU implementation
function idct!(X::AbstractArray{<:AbstractFloat}, dims=1:ndims(X))
    dims = [dims...]
    d  = size(X);
    r1 = [StepRange.(1,1,d)...]
    r2 = deepcopy(r1)

    cl = dst_scratch(d, typeof(X))
    for i in dims
        X2,X1 = dst_scratch(d, i)
        if d[i]>1
            if iseven(d[i])
                # Even sized
                ri    = r1[i]
                w     = _dct_scale(X, d, i)
                w    .= 1 ./ w
                w[1:1] ./= 2
                X1 .= X.*w
                ifft!(X1, i)
                r1[i] = 1:2:(d[i]-1)
                r2[i] = 1:1:Int(d[i]/2)
                X[r1...] .= real.(view(X1,r2...))
                r1[i] = 2:2:d[i]
                r2[i] = d[i]:-1:(Int(d[i]/2)+1)
                X[r1...] .= real.(view(X1,r2...))
                r1[i] = r2[i] = ri
            else
                # Odd sized
                ri    = r1[i]
                w     = _dct_scale(X, d, i)
                w    .= 1 ./ w
                X2[r2...] .= X.*w
                w     = reshape(conj.(w[2:end]), _along(length(d), i, d[i]-1))
                r2[i] = (d[i]*2):-1:(d[i]+2)
                r1[i] = 2:d[i]
                X2[r2...] = view(X,r1...).*w

                r2[i] = (d[i]+1):(d[i]+1)
                X2[r2...] .= 0
                r1[i] = ri
                ifft!(X2,i)
                X .= real.(view(X2,r1...))
                r2[i] = ri
            end
        end
    end
    dst_scratch(cl)
    return X
end


function _scale_pre!(X::Array{<:AbstractFloat}, dims=1:ndims(X), s1::Real=sqrt(2.))
    dims = [dims...]
    X  .*= (1 ./ sqrt(prod(2 .* size(X)[dims])))
    _scale_first!(X, s1, dims)
end

function _scale_post!(X::Array{<:AbstractFloat}, dims=1:ndims(X), s1::Real=1/sqrt(2.))
    dims = [dims...]
    X .*= (1 ./ sqrt(prod(2 .* size(X)[dims])))
    _scale_first!(X, s1, dims)
end

function _scale_first!(X::Array{<:AbstractFloat}, s1::Real, dims=1:ndims(X))
    dims = [dims...]
    if s1==1
        return X
    end
    d   = size(X)
    r   = [1:n for n in d]
    for i in dims
        ri   = r[i]
        r[i] = 1:1
        X[r...] .*= s1
        r[i] = ri
    end
    return X
end


#"""
#Testing DCT/IDCT
# """
#function dct_test!(X::Array{<:AbstractFloat},dims=1:ndims(X))
#    return _scale_post!(r2r!(X, REDFT10, dims), dims)
#end
#function idct_test!(X::Array{<:AbstractFloat},dims=1:ndims(X))
#    return r2r!(_scale_pre!(X, dims), REDFT01, dims)
#end
#function dct_test(X::Array{<:AbstractFloat}, dims=1:ndims(X))
#    return dct_test!(deepcopy(X),dims)
#end
#function idct_test(X::Array{<:AbstractFloat}, dims=1:ndims(X))
#    return idct_test!(deepcopy(X),dims)
#end


function _dct_scale(x, d, i)
    T      = Base.typename(typeof(x)).wrapper{Complex{eltype(x)}}
    d1     = ones(Int64,length(d))
    d1[i]  = d[i]
    w      = reshape(T(0:(d[i]-1)) .*= -(im*pi/(2*d[i])), d1...)
    w     .= exp.(w) ./ sqrt(2*d[i])
    w[1:1]./= sqrt(2)
    return w
end

function _dst_scale(x, d, i)
    T      = Base.typename(typeof(x)).wrapper{Complex{eltype(x)}}
    d1     = ones(Int64,length(d))
    d1[i]  = d[i]
    w      = reshape(T(1:(d[i])) .*= -(im*pi/(2*d[i])), d1...)
    w     .= .-exp.(w) ./ sqrt(2*d[i])
    return w
end

function _along(nd, i, n)
    d1    = ones(typeof(n),nd)
    d1[i] = n
    return (d1...,)
end



