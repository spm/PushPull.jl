using PushPull
using Test
using CUDA
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
end

