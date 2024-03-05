using ChainRulesCore
# Code could be improved using @thunk
# https://juliadiff.org/ChainRulesCore.jl/dev/rule_author/writing_good_rules.html

"""
Allows automatic differentiation to be applied to `pull`
"""
function ChainRulesCore.rrule(::typeof(pull), f₀, ϕ, s=Settings())
    f₁        = pull(f₀, ϕ, s)
    odim      =  a    -> (size(a)[1:3]..., 1, size(a)[4:end]...)
    summation = (a,b) -> sum(reshape(a,odim(a)).*b, dims = 5)
    pb(δ)     = (NoTangent(),
                 @thunk(push(unthunk(δ), ϕ, size(f₀)[1:3], s)),                      # f₀
                 @thunk(reshape(summation(unthunk(δ),pull_grad(f₀, ϕ, s)),size(ϕ))), # ϕ
                 NoTangent())                                                        # s
    return f₁, pb
end


"""
Allows automatic differentiation to be applied to `pull_grad`
"""
function ChainRulesCore.rrule(::typeof(pull_grad), f₀, ϕ, s=Settings())
    ∇f        = pull_grad(f₀, ϕ, s)
    odim      = δ     -> (size(δ)[1:4]..., 1, size(δ)[5:end]...)
    summation = (a,b) -> reshape(sum(reshape(a, odim(a)).*b, dims=(4, 6)), size(ϕ))
    pb(δ)     = (NoTangent(),
                 @thunk(push_grad(unthunk(δ), ϕ, size(f₀)[1:3], s)),      # f₀
                 @thunk(summation(unthunk(δ), pull_hess(f₀, ϕ, s))),      # ϕ
                 NoTangent())                                             # s
    return ∇f, pb
end


"""
Allows automatic differentiation to be applied to `push`
"""
function ChainRulesCore.rrule(::typeof(push), f₁, ϕ, dim, s=Settings())
    f₀        = push(f₁, ϕ, dim, s)
    odim      =  a    -> (size(a)[1:3]..., 1, size(a)[4:end]...)
    summation = (a,b) -> reshape(sum(reshape(a,odim(a)).*b, dims = 5),size(ϕ))
    pb(δ)     = (NoTangent(),
                 @thunk(pull(unthunk(δ), ϕ, s)),                     # f₁
                 @thunk(summation(f₁, pull_grad(unthunk(δ), ϕ, s))), # ϕ
                 NoTangent(),                                        # dim
                 NoTangent())                                        # s
    return f₀, pb
end


"""
Allows automatic differentiation to be applied to `affine_pull`
"""
function ChainRulesCore.rrule(::typeof(affine_pull), f₀, Aff, s=Settings())
    f₁        = affine_pull(f₀, Aff, s)
    pb(δ)     = (NoTangent(),
                 affine_push(δ, Aff, size(f₀)[1:3], s),              # f₀
                 NoTangent(),                                        # Aff - TBD
                 NoTangent())                                        # s
    return f₁, pb
end


"""
Allows automatic differentiation to be applied to `affine_push`
"""
function ChainRulesCore.rrule(::typeof(affine_push), f₁, Aff, dim, s=Settings())
    f₀        = affine_push(f₁, Aff, dim, s)
    pb(δ)     = (NoTangent(),
                 affine_pull(δ, Aff, s),                             # f₁
                 NoTangent(),                                        # Aff - TBD
                 NoTangent(),                                        # dim
                 NoTangent())                                        # s
    return f₀, pb
end

