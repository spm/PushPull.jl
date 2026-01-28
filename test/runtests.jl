using PushPull
using Test
include("utils.jl")

function ad_consistency(d::NTuple{3,Int64}, reg::Vector{<:AbstractFloat})
    bnd = [2 1 1; 1 2 1; 1 1 2] # Sliding boundary
    vx  = [1.2, 0.7, 1.5]
    u0  = randn(Float32,(d...,3))
    L   = registration_operator(vx, reg)
    #L   = PushPull.reduce2fit!(L, d[1:3],bnd)
    Lsp = sparsify(L, d[1:3])
    K  = greens(L, d)
    v  = mom2vel(u0, K)
    u1 = vel2mom(v, Lsp, bnd)
    return sum((u1[:].-u0[:]).^2)./sum(u0[:].^2)
end

function operator_consistency(d::NTuple{3,Int64}, reg::Vector{<:AbstractFloat})
    bnd = [2 1 1; 1 2 1; 1 1 2] # Sliding boundary
    vx  = [1.2, 0.7, 1.5]
    u0  = randn(Float32,(d...,3))
    L   = registration_operator(vx, reg)
   #L   = PushPull.reduce2fit!(L, d[1:3],bnd)
    Lsp = sparsify(L, d[1:3])

    K  = greens(L, d)
    v  = mom2vel(u0, K)
    u1 = vel2mom(v, Lsp, bnd)

    return sum((u1[:].-u0[:]).^2)./sum(u0[:].^2)
end


@testset "PushPull.jl" begin
    # Write your tests here.

    ##################################
    # PUSH, PULL & PULL_GRAD
    ##################################
    # Comparing numerical vs analytic gradients needs a slightly
    # high tolerance.
    tol   = 1e-3

    d     = (8,7,1)
    c     = 2
    f1    = randn(Float32,(d...,c))
    f2    = randn(Float32,(d...,c))
    phi   = randn(Float32,(d...,3))
    phi .+= PushPull.id(phi)
    sett  = PushPull.Settings((3,3,3),(0,1,2),1)

    # Disabled because gradients are less predictable with low-degree
    # interpolation. This may be partly due to ambiguity in rounding
    # conventions.
    # Note that Julia's rounding convention is especially bizarre:
    # round.(Float64.(1:10) .- 5.5)
   #sett  = PushPull.Settings((1,1,1),(0,1,2),1)
   #@test test_grad(θ -> sum((pull(f1, θ,  sett) .- f2).^2),phi) < 5e-2
   #@test test_grad(θ -> sum((pull(θ,phi,  sett) .- f2).^2),f1)  < tol
   #@test test_grad(θ -> sum((push(f1, θ,d,sett) .- f2).^2),phi) < 5e-2
   #@test test_grad(θ -> sum((push(θ,phi,d,sett) .- f2).^2),f1)  < 5e-2

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


    # Testing batch-size>1 
    bs    = 2
    f1    = randn(Float32,(d...,c,bs))
    f2    = randn(Float32,(d...,c,bs))
    phi   = randn(Float32,(d...,3,bs))
    phi .+= PushPull.id(phi)

    # CPU
    sett  = PushPull.Settings((3,3,3),(0,1,2),1)
    @test test_grad(θ -> sum((pull(f1, θ,  sett) .- f2).^2),phi) < tol
    @test test_grad(θ -> sum((pull(θ,phi,  sett) .- f2).^2),f1)  < tol    #
    @test test_grad(θ -> sum((push(f1, θ,d,sett) .- f2).^2),phi) < tol    #
    @test test_grad(θ -> sum((push(θ,phi,d,sett) .- f2).^2),f1)  < tol    #

    # Tests of the push/pull adjoint
    tol = 1e-5
    @test abs(sum(f1.*pull(f2,phi,sett)) - sum(push(f1,phi,d,sett).*f2)) < tol
end
nothing

