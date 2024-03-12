using PushPull
using Test

function operator_consistency(d::NTuple{3,Int64}, reg::Vector{<:AbstractFloat}, cu::Bool)
    bnd = [2 1 1; 1 2 1; 1 1 2] # Sliding boundary
    u0  = randn(Float32,(d...,3))
    L   = registration_operator([1,1,1], reg)
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

tol = 1e-4
@testset "PushPull.jl" begin
    # Write your tests here.
    @test operator_consistency((8,7,1),[1e-3, 0.,0.,0.], true) < tol
   #@test operator_consistency((8,7,1),[1e-3, 1.,0.,0.], true) < tol
   #@test operator_consistency((8,7,1),[1e-3, 1.,1.,0.], true) < tol
   #@test operator_consistency((8,7,1),[1e-3, 1.,1.,9.], true) < tol

    @test operator_consistency((32,32,32),[1e-3, 0.,0.,0.], true) < tol
    @test operator_consistency((32,32,32),[1e-3, 1.,0.,0.], true) < tol
    @test operator_consistency((32,32,32),[1e-3, 1.,1.,0.], true) < tol
    @test operator_consistency((32,32,32),[1e-3, 1.,1.,9.], true) < tol

    @test operator_consistency((8,7,1),[1e-3, 0.,0.,0.], false) < tol
   #@test operator_consistency((8,7,1),[1e-3, 1.,0.,0.], false) < tol
   #@test operator_consistency((8,7,1),[1e-3, 1.,1.,0.], false) < tol
   #@test operator_consistency((8,7,1),[1e-3, 1.,1.,9.], false) < tol
    @test operator_consistency((32,32,32),[1e-3, 0.,0.,0.], false) < tol
    @test operator_consistency((32,32,32),[1e-3, 1.,0.,0.], false) < tol
    @test operator_consistency((32,32,32),[1e-3, 1.,1.,0.], false) < tol
    @test operator_consistency((32,32,32),[1e-3, 1.,1.,9.], false) < tol
end

