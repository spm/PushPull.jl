
"""
    compose!(ϕ::T{Float32,4}, θ::T{Float32,4}) where {T:<AbstractArray}

Requirements
* `ϕ` must have three channels
* `θ` must have three channels
"""
function compose(ϕ::T, θ::T) where {T<:AbstractArray{Float32,4}}
    @assert(size(ϕ,4) == 3,          "`ϕ` must have three channels")
    @assert(size(θ,4) == 3,          "`θ` must have three channels")
    ψ = zero(θ)
    compose!(ψ, ϕ, θ)
    return ψ
end


"""
    compose!(ψ::T{Float32,4}, ϕ::T{Float32,4}, θ::T{Float32,4}) where {T:<AbstractArray}

Requirements
* `ϕ` must have three channels
* `θ` must have three channels
* `ψ` must have three channels
* `ψ` and `θ` must have the same dimensions
"""
function compose!(ψ::T, ϕ::T, θ::T) where {T<:AbstractArray{Float32,4}}
    @assert(size(ϕ,4) == 3,           "`ϕ` must have three channels")
    @assert(size(θ,4) == 3,           "`θ` must have three channels")
   #@assert(size(ψ,4) == 3,           "`ψ` must have three channels")
    @assert(all(size(ψ) .== size(θ)), "`ψ` and `θ` must have the same dimensions")
    deg = (1,1,1)
    bnd = [2 1 1; 1 2 1; 1 1 2]
    for dim=1:3
        sett = Settings(deg, (bnd[:,dim]...,),true)
        ψ[:,:,:,dim] .= pull(ϕ[:,:,:,dim], θ, sett)
    end
    return ψ
end

function Exp(v::AbstractArray{Float32,4})
    Id = id(v)
    u  = v/(2^8)
    for k=1:8
        u .= compose(u, Id .+ u)
    end
    u .+= Id
    return u
end

nothing

