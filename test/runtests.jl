using PushPull
using Test
using CUDA
using Flux

function gpu_cpu_K(d::NTuple{3,Int64}, reg::Vector{<:AbstractFloat})
    vx = [1.2, 0.7, 1.5]
    u  = randn(Float32,(d...,3))
    L  = registration_operator(vx, reg)
    Kc = greens(L, d)
    vc = mom2vel(u, Kc)

    u  = CuArray(u)
    L  = CuArray(L)
    Kg = greens(L,d)
    vg = mom2vel(u, Kg)

    return sum((vc - Array(vg)).^2)/sum(vc.^2)
end

function gpu_cpu_L(d::NTuple{3,Int64}, reg::Vector{<:AbstractFloat})
    vx  = [1.2, 0.7, 1.5]
    bnd = [2 1 1; 1 2 1; 1 1 2]
    bnd = [0 0 0; 0 0 0; 0 0 0];
    v   = randn(Float32,(d...,3))
    L   = registration_operator(vx, reg)
    Lsp = sparsify(L, d[1:3])

    uc  = vel2mom(v, Lsp, bnd)
    v   = CuArray(v)
    ug  = vel2mom(v, Lsp, bnd)
    return sum((uc - Array(ug)).^2)/sum(uc.^2)
end

function operator_consistency(d::NTuple{3,Int64}, reg::Vector{<:AbstractFloat}, cu::Bool)
    bnd = [2 1 1; 1 2 1; 1 1 2] # Sliding boundary
    vx  = [1.2, 0.7, 1.5]
    u0  = randn(Float32,(d...,3))
    L   = registration_operator(vx, reg)
   #L   = PushPull.reduce2fit!(L, d[1:3],bnd)
    Lsp = sparsify(L, d[1:3])

    if cu
        L  = CuArray(L)
        u0 = CuArray(u0)
    end

    K  = greens(L, d);
    v  = mom2vel(u0, K);
    u1 = vel2mom(v, Lsp, bnd);

    return sum((u1[:].-u0[:]).^2)./sum(u0[:].^2)
end

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



@testset "PushPull.jl" begin
    # Write your tests here.

    tol = 1e-8
    d = (8,7,1)
    @test gpu_cpu_L(d, [1e-3, 1.,0.,0.]) < tol
    @test gpu_cpu_L(d, [1e-3, 1.,9.,0.]) < tol
    @test gpu_cpu_L(d, [1e-3, 1.,9.,1.]) < tol

    d = (32,31,30)
    @test gpu_cpu_L(d, [1e-3, 1.,0.,0.]) < tol
    @test gpu_cpu_L(d, [1e-3, 1.,9.,0.]) < tol
    @test gpu_cpu_L(d, [1e-3, 1.,9.,1.]) < tol

    # Bigger tolerance because a bit less stable
    tol = 1e-5
    d = (8,7,1)
    @test gpu_cpu_K(d, [1e-3, 1.,0.,0.]) < tol
    @test gpu_cpu_K(d, [1e-3, 1.,9.,0.]) < tol
    @test gpu_cpu_K(d, [1e-3, 1.,9.,1.]) < tol

    d = (32,31,30)
    @test gpu_cpu_K(d, [1e-3, 1.,0.,0.]) < tol
    @test gpu_cpu_K(d, [1e-3, 1.,9.,0.]) < tol
    @test gpu_cpu_K(d, [1e-3, 1.,9.,1.]) < tol

    tol = 1e-8
    d = (8,7,1)
    @test operator_consistency(d,[1e-3, 0.,0.,0.], true) < tol
    @test operator_consistency(d,[1e-3, 1.,0.,0.], true) < tol
    @test operator_consistency(d,[1e-3, 1.,9.,0.], true) < tol
    @test operator_consistency(d,[1e-3, 1.,9.,1.], true) < tol

    d = (32,31,30)
    @test operator_consistency(d,[1e-3, 0.,0.,0.], true) < tol
    @test operator_consistency(d,[1e-3, 1.,0.,0.], true) < tol
    @test operator_consistency(d,[1e-3, 1.,9.,0.], true) < tol
    @test operator_consistency(d,[1e-3, 1.,9.,1.], true) < tol

    d = (8,7,1)
    @test operator_consistency(d,[1e-3, 0.,0.,0.], false) < tol
    @test operator_consistency(d,[1e-3, 1.,0.,0.], false) < tol
    @test operator_consistency(d,[1e-3, 1.,9.,0.], false) < tol
    @test operator_consistency(d,[1e-3, 1.,9.,1.], false) < tol

    d = (32,31,30)
    @test operator_consistency(d,[1e-3, 0.,0.,0.], false) < tol
    @test operator_consistency(d,[1e-3, 1.,0.,0.], false) < tol
    @test operator_consistency(d,[1e-3, 1.,9.,0.], false) < tol
    @test operator_consistency(d,[1e-3, 1.,9.,1.], false) < tol

    d = (2,3,4)
    @test operator_consistency(d,[1e-3, 1.,9.,1.], false) < tol

    d     = (8,7,1)
    c     = 2

    f1    = randn(Float32,(d...,c))
    f2    = randn(Float32,(d...,c))
    phi   = randn(Float32,(d...,3))
    phi .+= PushPull.id(d,false)

    tol   = 1e-3 
    sett  = PushPull.Settings((1,1,1),(0,1,2),1)
    @test test_grad(θ -> sum((pull(f1, θ,  sett) .- f2).^2),phi) < 1e-2
    @test test_grad(θ -> sum((pull(θ,phi,  sett) .- f2).^2),f1)  < tol
    @test test_grad(θ -> sum((push(f1, θ,d,sett) .- f2).^2),phi) < 5e-2
    @test test_grad(θ -> sum((push(θ,phi,d,sett) .- f2).^2),f1)  < 5e-2

    sett  = PushPull.Settings((2,2,2),(0,1,2),1)
    @test test_grad(θ -> sum((pull(f1, θ,  sett) .- f2).^2),phi) < tol
    @test test_grad(θ -> sum((pull(θ,phi,  sett) .- f2).^2),f1)  < tol
    @test test_grad(θ -> sum((push(f1, θ,d,sett) .- f2).^2),phi) < tol
    @test test_grad(θ -> sum((push(θ,phi,d,sett) .- f2).^2),f1)  < tol

    sett  = PushPull.Settings((3,3,3),(0,1,2),1)
    @test test_grad(θ -> sum((pull(f1, θ,  sett) .- f2).^2),phi) < tol
    @test test_grad(θ -> sum((pull(θ,phi,  sett) .- f2).^2),f1)  < tol
    @test test_grad(θ -> sum((push(f1, θ,d,sett) .- f2).^2),phi) < tol
    @test test_grad(θ -> sum((push(θ,phi,d,sett) .- f2).^2),f1)  < tol

    g2 = pull_grad(f2, phi,  sett)
    @test test_grad(θ -> sum((pull_grad(f1, θ,  sett) .- g2).^2),phi) < tol

    f1   = CuArray(f1)
    f2   = CuArray(f2)
    phi  = CuArray(phi)

    sett = PushPull.Settings((1,1,1),(0,1,2),1)
    @test test_grad(θ -> sum((pull(f1, θ,  sett) .- f2).^2),phi) < 1e-2
    @test test_grad(θ -> sum((pull(θ,phi,  sett) .- f2).^2),f1)  < tol
    @test test_grad(θ -> sum((push(f1, θ,d,sett) .- f2).^2),phi) < 5e-2
    @test test_grad(θ -> sum((push(θ,phi,d,sett) .- f2).^2),f1)  < 5e-2

    sett  = PushPull.Settings((2,2,2),(0,1,2),1)
    @test test_grad(θ -> sum((pull(f1, θ,  sett) .- f2).^2),phi) < tol
    @test test_grad(θ -> sum((pull(θ,phi,  sett) .- f2).^2),f1)  < tol
    @test test_grad(θ -> sum((push(f1, θ,d,sett) .- f2).^2),phi) < tol
    @test test_grad(θ -> sum((push(θ,phi,d,sett) .- f2).^2),f1)  < tol

    sett  = PushPull.Settings((3,3,3),(0,1,2),1)
    @test test_grad(θ -> sum((pull(f1, θ,  sett) .- f2).^2),phi) < tol
    @test test_grad(θ -> sum((pull(θ,phi,  sett) .- f2).^2),f1)  < tol
    @test test_grad(θ -> sum((push(f1, θ,d,sett) .- f2).^2),phi) < tol
    @test test_grad(θ -> sum((push(θ,phi,d,sett) .- f2).^2),f1)  < tol

    bs    = 2
    f1    = randn(Float32,(d...,c,bs))
    f2    = randn(Float32,(d...,c,bs))
    phi   = randn(Float32,(d...,3,bs))
    phi .+= PushPull.id(d,false)

    sett  = PushPull.Settings((3,3,3),(0,1,2),1)
    @test test_grad(θ -> sum((pull(f1, θ,  sett) .- f2).^2),phi) < tol
    @test test_grad(θ -> sum((pull(θ,phi,  sett) .- f2).^2),f1)  < tol
    @test test_grad(θ -> sum((push(f1, θ,d,sett) .- f2).^2),phi) < tol
    @test test_grad(θ -> sum((push(θ,phi,d,sett) .- f2).^2),f1)  < tol

    f1   = CuArray(f1)
    f2   = CuArray(f2)
    phi  = CuArray(phi)
    @test test_grad(θ -> sum((pull(f1, θ,  sett) .- f2).^2),phi) < tol
    @test test_grad(θ -> sum((pull(θ,phi,  sett) .- f2).^2),f1)  < tol
    @test test_grad(θ -> sum((push(f1, θ,d,sett) .- f2).^2),phi) < tol
    @test test_grad(θ -> sum((push(θ,phi,d,sett) .- f2).^2),f1)  < tol

    g2 = pull_grad(f2, phi,  sett)
    @test test_grad(θ -> sum((pull_grad(f1, θ,  sett) .- g2).^2),phi) < tol
end
nothing

