using Zygote

function test_grad(fun,ϕ)

    function numerical_grad(fun, θ₀)
        E₀ = fun(θ₀)
        g  = zero(θ₀)
        θ  = deepcopy(θ₀)
        ϵ  = 0.01
        for i=1:length(θ₀)
            # Used i:i because scalar indexing of CUDA.jl
            # arrays is problematic.
            θ[i:i] .+= ϵ
            E₊         = fun(θ)
            θ[i:i] .-= 2ϵ
            E₋         = fun(θ)
            g[i:i]    .= (E₊ - E₋)/(2ϵ)
            θ[i:i]    .= θ₀[i:i]
        end
        return g
    end

    g0 = gradient(fun,ϕ)[1]
    return sum((g0-numerical_grad(fun,ϕ)).^2)/sum(g0.^2)
end

