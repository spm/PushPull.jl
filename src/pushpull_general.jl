
"""
## Summary

  `struct Main.PushPull.Settings <: Any`

Encodes the push-pull settings.

##  Fields
* `deg :: Tuple{Integer,Integer,Integer}` - Degree of interpolation 0..4 along each dimention
* `bnd :: Tuple{Integer,Integer,Integer}` - Boundary conditions 0=wrap/1=reflect/2=?? along each dimension
* `ext :: Integer`                        - Extrapolate missing data or not

"""
struct Settings
    deg::NTuple{3,UInt64}
    bnd::Array{NTuple{3,Int32},1}
    ext::Int32
    function Settings(deg::Union{Integer,NTuple{3,<:Integer}} = UInt64.((1,1,1)),
                      bnd::Union{Integer,
                                 NTuple{3,<:Integer},
                                 Vector{<:NTuple{3,Integer}},
                                 Matrix{<:Integer}} = [Int32.((1,1,1))],
                      ext::Integer = true)

        # Can be 0,1,2,3 or 4 so constrain
        if isa(deg,Integer)
            deg = (deg,deg,deg)
        end
        deg     = UInt64.(max.(min.(deg,4),0))

        # Boundary conditions can be mixed
        allowed = (0,1,2)
        if bnd isa Integer
            @assert(bnd ∈ allowed)
            bnd  = Int32(bnd)
            bnd1 = [(bnd,bnd,bnd)]
        elseif bnd isa Matrix
            @assert(size(bnd,1)==3)
            @assert(all(bnd .∈ (allowed,)))
            bnd1 = Vector{NTuple{3,Int32}}(undef,size(bnd,2))
            for i=1:size(bnd,2)
                bnd1[i] = (Int32.(bnd[:,i])...,)
            end
        elseif bnd isa Vector{<:NTuple}
            bnd1 = Vector{NTuple{3,Int32}}(undef,length(bnd))
            for i=1:length(bnd)
                @assert(all(bnd[i] .∈ (allowed,)))
                bnd1[i] = Int32.(bnd[i])
            end
        else # bnd isa NTuple
            @assert(all(bnd .∈ (allowed,)))
            bnd1 = [Int32.(bnd)]
        end

        # true / false
        ext = Int32(max(min(ext,1),0))
        new(deg,bnd1,ext)
    end
end


function sett_chan(sett::Settings, c::Integer)
    if length(sett.bnd)==1
        return Settings
    else
        return Settings(sett.deg, sett.bnd[c], sett.ext)
    end
end

#function show(sett::Settings)
#    print("deg: ", sett.deg, "\n")
#    print("bnd: ", sett.bnd, "\n")
#    print("ext: ", sett.ext, "\n")
#end
#using Printf
#function show(sett::Settings)
#    @printf("deg: %d,%d,%d\nbnd: %d,%d,%d\next: %d\n",Int32.(sett.deg)..., Int32.(sett.bnd)..., Int32(sett.ext))
#end

function id(v::AbstractArray{Float32,4})
    return id(size(v)[1:3]; gpu=isa(v,CuArray))
end

function id(d1::NTuple{3,Integer}; gpu::Integer = false, affine::Matrix{<:Real} = [1. 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1])
    if gpu==0
        x   = Float32.(1:d1[1])
        y   = Float32.(1:d1[2])'
        z   = reshape(Float32.(1:d1[3]),(1,1,d1[3]))
        phi = zeros(Float32,(d1...,3))
    else
        x   =         CuArray{Float32}(1:d1[1])              
        y   = reshape(CuArray{Float32}(1:d1[2]), (1,d1[2],1))
        z   = reshape(CuArray{Float32}(1:d1[3]), (1,1,d1[3]))
        phi = CUDA.zeros(Float32,(d1...,3))
    end
    if affine==nothing
        phi[:,:,:,1] .= x
        phi[:,:,:,2] .= y
        phi[:,:,:,3] .= z
    else
        A = affine
        phi[:,:,:,1] .= Float32.(A[1,1].*x .+ A[1,2].*y .+ A[1,3].*z .+ A[1,4])
        phi[:,:,:,2] .= Float32.(A[2,1].*x .+ A[2,2].*y .+ A[2,3].*z .+ A[2,4])
        phi[:,:,:,3] .= Float32.(A[3,1].*x .+ A[3,2].*y .+ A[3,3].*z .+ A[3,4])
    end
    return phi
end

