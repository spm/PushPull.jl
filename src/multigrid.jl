VolType{N} = Union{CuArray{Float32,N}, Array{Float32,N}}

dim(v) = size(v)[1:3]

struct HessType<:Any
    H::VolType
    L::KernelType
    bnd::Array{Int32,2}
end

PyramidType = Dict{NTuple{3, Int64}, HessType}

"""

    restrict(f1::VolType, d0::NTuple{3,Int}=Int64.(ceil.(dim(f1)./2)), deg::Integer=1)

Restriction (downsampling to a coarser grid) in 3D.

"""
function restrict(f1::VolType, d0::NTuple{3,Int}=Int64.(ceil.(dim(f1)./2)), deg::Integer=2)
    d1  = dim(f1)
    zm  = d0./d1;
    os  = (1 .- zm)./2
    Mat = Float32.([zm[1] 0. 0. os[1]; 0. zm[2] 0. os[2];  0. 0. zm[3] os[3]; 0. 0. 0. 1.])
    f0  = affine_push(f1, Mat, d0, Settings(deg,(1,1,1),true))

    c1  = typeof(f1)(undef,(size(f1)[1:3]...,1))
    c1 .= 1f0 #Float32(prod(d0)/prod(d1))
    c0  = affine_push(c1, Mat, d0, Settings(deg,(1,1,1),true))
    f0 ./= c0

    return f0
end

"""

    prolong(f0::VolType, d1::NTuple{3,Int}=dim(f0)*2, deg::Integer=1)

Prolongation (interpolating to a finer grid) in 3D.

"""
function prolong(f0::VolType, d1::NTuple{3,Int}=2dim(f0), deg::Integer=2)
    d0  = dim(f0)
    zm  = d0./d1;
    os  = (1 .- zm)./2
    Mat = Float32.([zm[1] 0. 0. os[1]; 0. zm[2] 0. os[2];  0. 0. zm[3] os[3]; 0. 0. 0. 1.])
    bnd = [2 1 1; 1 2 1; 1 1 2] # Sliding boundaries
    f1  = affine_pull(f0, Mat, d1, Settings(deg,bnd,true))
    return f1
end

"""

    Hv!(v::VolType, H::VolType, u::VolType)

Multiplying `v` by the Hessian (`H`).

"""

function Hv!(v::VolType, H::VolType, u::VolType=zero(v))::VolType
    @assert(all(dim(H)  .== dim(v)))
    @assert(all(size(v) .== size(u)))
    @assert(ndims(v)==4 && ndims(H)==4)
    dv = size(v,4)
    dh = size(H,4)
    @assert(dh==1 || dh==dv || dh == Int((dv+1)*dv/2))

    if false #size(v,4) == 3 # Special case
        v1   = view(v,:,:,:,1)
        v2   = view(v,:,:,:,2)
        v3   = view(v,:,:,:,3)
        u1   = view(u,:,:,:,1)
        u2   = view(u,:,:,:,2)
        u3   = view(u,:,:,:,3)
        if size(H,4) >= 3
            h11  = view(H,:,:,:,1)
            h22  = view(H,:,:,:,2)
            h33  = view(H,:,:,:,3)
            if size(H,4) == 6
                h12  = view(H,:,:,:,4)
                h13  = view(H,:,:,:,5)
                h23  = view(H,:,:,:,6)
                u1 .+= h11.*v1 .+ h12.*v2 .+ h13.*v3
                u2 .+= h12.*v1 .+ h22.*v2 .+ h23.*v3
                u3 .+= h13.*v1 .+ h23.*v2 .+ h33.*v3
            elseif size(H,4) == 3
                u1 .+= h11.*v1
                u2 .+= h22.*v2
                u3 .+= h33.*v3
            else
                error()
            end
        elseif size(H,4) == 1
            h11 = view(H,:,:,:,1)
            u1 .+= h11.*v1
            u2 .+= h11.*v2
            u3 .+= h11.*v3
        else
            error()
        end
        return u
    else # General case
        if dh==1
            h   = view(H,:,:,:,1)
            for i=1:dv
                view(u,:,:,:,i) .+= h.*view(v,:,:,:,i)
            end
        elseif dh==dv || dh==Int((dv+1)*dv/2)
            for i=1:dv
                view(u,:,:,:,i) .+= view(H,:,:,:,i).*view(v,:,:,:,i)
            end
            if dh==Int((dv+1)*dv/2)
                ii = dv
                for i=1:dv, j=i+1:dv
                    ii += 1
                    h   = view(H,:,:,:,ii)
                    view(u,:,:,:,i) .+= h.*view(v,:,:,:,j)
                    view(u,:,:,:,j) .+= h.*view(v,:,:,:,i)
                end
            end
        end
    end
    return u
end


"""
    HLv(v::VolType, H::Dict)

Multiplying `v` by the full Hessian (`H + L`).

"""
function HLv(v::VolType, HL::PyramidType)::VolType
    hl   = HL[dim(v)]
    u    = vel2mom(v, hl.L, hl.bnd)
    return Hv!(v, hl.H, u)
end


"""

    relax!(g::VolType, H::Dict, nit::Int=2, u::VolType=zero(g))

Gauss-Siedel relaxation to update `u` from gradients `g`, and a Hessian
`H[dim(g)]`.

See `https://en.wikipedia.org/wiki/Multigrid_method` for more information.

"""
function relax!(g::VolType, HL::Dict, nit::Int=2, v::VolType=zero(g))::VolType
    d   = dim(g)
    H   = HL[d].H
    L   = HL[d].L
    bnd = HL[d].bnd
    relax!(g, H, L, bnd, nit, v)
    return v
end


"""

    hessian_pyramid(h::VolType, vx::Vector{Float32}, reg::Vector{<:Real})

Create a Hessian pyramid (`HL`) from `h`, with regularisation based
on voxel sizes `vx` and regularisation parameters `reg`.

`HL` is a dictionary of Hessians at different spatial scales, with `HL[dim(g)]`
giving an appropriate Hessian for scale `dim(g)`.

"""
function hessian_pyramid(h::VolType,
                         vx::Vector{<:Real}=[1f0,1f0,1f0],
                         reg::Vector{<:Real}=[1f-3, 1, 0, 0])::PyramidType
    vx    = Float32.(vx)
    reg   = Float32.(reg)
    bnd   = [2 1 1; 1 2 1; 1 1 2]
    regop(d,vx,reg,vx0) = sparsify(reduce2fit(registration_operator(vx, reg,vx0), d, bnd), d)
    d0    = d = dim(h)
    HL    = PyramidType()
    HL[d] = HessType(h,regop(d,vx,reg,vx),bnd)
    while any(d .> 1)
        h     = restrict(h)
        h[:,:,:,1:3] .*= 1.0001f0
        d     = dim(h)
        HL[d] = HessType(h , regop(d,Float32.(vx.*d0./d),reg,vx), bnd)
    end
    return HL
end


"""

    vcycle!(v::VolType, g::VolType, HL::PyramidType; nit_pre::Integer=4, nit_post::Integer=4)

Run a V-cycle to refine the estimate of `v` from gradients `g` and Hessian pyramid `HL`.

See `https://en.wikipedia.org/wiki/Multigrid_method` for more information.

"""
function vcycle!(v::VolType, g::VolType, HL::PyramidType; nit_pre::Integer=4, nit_post::Integer=4)
    if all(dim(v).==1)
        relax!(g, HL, nit_pre, v)
    else
        relax!(g, HL, nit_pre, v)
        g1 = restrict(g .- HLv(v, HL))
        v1 = zero(g1)
        vcycle!(v1,g1,HL; nit_pre=nit_pre, nit_post=nit_post)
        v .+= prolong(v1, dim(v))
        relax!(g, HL, nit_post, v)
    end
    return v
end

function vcycle_verb!(v::VolType, g::VolType, HL::PyramidType; nit_pre::Integer=4, nit_post::Integer=4)
    function res(v,HL,g,txt)
        u   = HLv(v, HL)
        pre = repeat("    ", Int.(round(log2(maximum(size(g)[1:3])))))
        print(pre, size(g)[1:3], " ", txt, " ", sum((u.-g).^2),"\n")
    end

    if all(dim(v).==1)
        res(v,HL,g,"pre  ")
        relax!(g, HL, nit_pre, v)
        res(v,HL,g,"post ")
    else
        res(v,HL,g,"pre  ")
        relax!(g, HL, nit_pre, v)
        res(v,HL,g,"relax")
        g1 = restrict(g .- HLv(v, HL))
        v1 = zero(g1)
        vcycle_verb!(v1,g1,HL; nit_pre=nit_pre, nit_post=nit_post)
        v .+= prolong(v1, dim(v))
        res(v,HL,g,"prolo")
        relax!(g, HL, nit_post, v)
        res(v,HL,g,"post ")
    end
    return v
end

function fcycle!(v::VolType, g::VolType, HL::PyramidType; nit_pre::Integer=4, nit_post::Integer=4)
    if all(dim(v).==1)
        relax!(g, HL, nit_pre, v)
    else
        relax!(g, HL, nit_pre, v)
        g1 = restrict(g .- HLv(v, HL))
        v1 = zero(g1)
        fcycle!(v1,g1,HL; nit_pre=nit_pre, nit_post=nit_post)
        v .+= prolong(v1, dim(v))
        relax!(g, HL, nit_post, v)
        g1 = restrict(g .- HLv(v, HL))
        v1 = zero(g1)
        vcycle!(v1,g1,HL; nit_pre=nit_pre, nit_post=nit_post)
        v .+= prolong(v1, dim(v))
        relax!(g, HL, nit_post, v)
    end
    return v
end


function invert_def(phi::T)::T where T<:VolType
    d    = size(phi)
    if length(d)>4
        iphi = zero(phi)
        for i in CartesianIndices(d[5:end])
            iphi[:,:,:,:,i] .= invert_def(phi[:,:,:,:,i])
        end
        return iphi
    else
        Id   = id(d[1:3]; gpu=~isa(phi,Array))
        sett = Settings(1, [2 1 1;1 2 1;1 1 2], 0)
        g    = push(Id, phi, d[1:3], sett)
        o    = typeof(g)(undef,(d[1:3]...,1))
        o   .= 1
        h    = push(o, phi, d[1:3], Settings(1,1,0))
        g  .-= h.*Id
        H    = typeof(g)(undef,(d[1:3]...,3))
        H[:,:,:,1:3].=h
        HL   = hessian_pyramid(H,[1f0,1f0,1f0],[0., 0.01, 0.1, 0.01])
        iphi = zero(g)
        vcycle!(iphi, g, HL; nit_pre=2, nit_post=2)
        iphi .+= Id
        return iphi
    end
end

