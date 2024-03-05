
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
    deg::NTuple{3,Integer}
    bnd::NTuple{3,Integer}
    ext::Integer
    function Settings(deg = (1,1,1), bnd = (1,1,1), ext = true)
        deg = UInt64.(max.(min.(deg,4),0))  # Can be 0,1,2,3 or 4 so constrain
        bnd = Int32.(bnd)
        if !all(in.(bnd,((0,1,2),))) error("inappropriate boundary condition"); end
        ext = Int32(max(min(ext,1),0))      # true / false
        new(deg,bnd,ext)
    end
end

#using Printf
#function show(sett::Settings)
#    @printf("deg: %d,%d,%d\nbnd: %d,%d,%d\next: %d\n",Int32.(sett.deg)..., Int32.(sett.bnd)..., Int32(sett.ext))
#end

function id(v::AbstractArray{Float32,4})
    return id(size(v)[1:3], isa(v,CuArray))
end

function id(d1::NTuple{3,Integer}, gpuflag::Integer = true, A::Matrix{<:Real} = [1. 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1])
    if gpuflag==0
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
    phi[:,:,:,1] .= Float32.(A[1,1].*x .+ A[1,2].*y .+ A[1,3].*z .+ A[1,4])
    phi[:,:,:,2] .= Float32.(A[2,1].*x .+ A[2,2].*y .+ A[2,3].*z .+ A[2,4])
    phi[:,:,:,3] .= Float32.(A[3,1].*x .+ A[3,2].*y .+ A[3,3].*z .+ A[3,4])
    return phi
end

