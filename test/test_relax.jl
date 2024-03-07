#include("relax_sparse.jl")
#include("operator.jl")
#include("relax.jl")
using PushPull
using CUDA

bnd  = ones(Int32,3,3)
for i=1:3
    bnd[i,i]=2
end

d = (256,256,256);
d = (126,128,127)
#d = (64,64,64)
#d = (5,5,5)

u0     = randn(Float32,(d...,3));
u0     = CuArray(u0)

print("Operator ", d, "\n")
CUDA.@time L   = registration_operator([1,1,1],[0,1,0,1]);
CUDA.@time L   = registration_operator([1,1,1],[0,1,0,1]);

#L     = reduce2fit!(L, d[1:3]);
print("\nSparsify\n")
CUDA.@time Lsp = sparsify(L, d[1:3]);
CUDA.@time Lsp = sparsify(L, d[1:3]);

print("\nGreens\n")
L = CuArray(L);
CUDA.@time K   = greens(L,d);
CUDA.unsafe_free!.(K)
CUDA.@time K   = greens(L,d);

print("\nmom2vel\n")
CUDA.@time v   = mom2vel(u0,K);
CUDA.unsafe_free!(v)
CUDA.@time v   = mom2vel(u0,K);

print("\nvel2mom\n")
CUDA.@time u1  = vel2mom(v,Lsp,bnd);
CUDA.unsafe_free!(u1)
CUDA.@time u1  = vel2mom(v,Lsp,bnd);

print(sum(u0[:].^2), " ", sum(v[:].^2), " ", sum((u1[:].-u0[:]).^2), "\n")

CUDA.unsafe_free!(v);
CUDA.unsafe_free!(u1);
Lt = Array(L);
CUDA.unsafe_free!(L);
ut = Array(u0);
CUDA.unsafe_free!(u0);
CUDA.unsafe_free!.(K)
CUDA.reclaim()


if true
print("=== CPU ===\n")
u0 = ut
CUDA.reclaim()

print("Greens\n")
L = Lt;
@time K = greens(L,d)
@time K = greens(L,d)

print("mom2vel\n")
@time v = mom2vel(u0,K);
@time v = mom2vel(u0,K);

print("vel2mom\n")
@time u1 = vel2mom(v,Lsp,bnd);
@time u1 = vel2mom(v,Lsp,bnd);

print(sum(u0[:].^2), " ", sum(v[:].^2), " ", sum((u1[:]-u0[:]).^2), "\n")
end

#nit = 4;
#L   = registration_operator([1,1,1],[0,1,0,1])

#@time vd = Relax.relax(g, h, L, bnd, nit);

#L   = registration_operator([1,1,1],[1,0,0,0])
#kernel = sparsify(L,d[1:3]);
#print(kernel.stride,"\n")
#@time vs = relax!(g, h, kernel, bnd, 1);
#@time vs = relax!(g, h, kernel, bnd, nit);

#L   = registration_operator([1,1,1],[0,1,0,0])
#kernel = sparsify(L,d[1:3]);
#print(kernel.stride,"\n")
#@time vs = relax!(g, h, kernel, bnd, 1);
#@time vs = relax!(g, h, kernel, bnd, nit);

#L   = registration_operator([1,1,1],[0,1,0,1])
#kernel = sparsify(L,d[1:3]);
#print(kernel.stride,"\n")
#@time vs = relax!(g, h, kernel, bnd, 1);
#@time vs = relax!(g, h, kernel, bnd, nit);

#L   = registration_operator([1,1,1],[0,1,1,0])
#kernel = sparsify(L,d[1:3]);
#print(kernel.stride,"\n")
#@time vs = relax!(g, h, kernel, bnd, 1);
#@time vs = relax!(g, h, kernel, bnd, nit);

#L   = registration_operator([1,1,1],[0,1,1,1])
#kernel = sparsify(L,d[1:3]);
#print(kernel.stride,"\n")
#@time vs = relax!(g, h, kernel, bnd, 1);
#@time vs = relax!(g, h, kernel, bnd, nit);



