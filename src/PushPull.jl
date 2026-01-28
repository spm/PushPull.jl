module PushPull

include("locations.jl")
include("pushpull_general.jl")
include("pushpull_init.jl")
include("pushpull_cpu.jl")
include("pushpull_pb.jl")
include("dst.jl")
include("operator.jl")
include("operator_sparse.jl")
include("multigrid.jl")
include("denoise3d_cpu.jl")

if false
    print("\n ### Using CUDA ### \n")
    using CUDA.CUFFT
    include("pushpull_gpu.jl")
    include("operator_sparse_gpu.jl")
    include("lbessi_gpu.jl")
    include("denoise3d_gpu.jl")
    include("init_gpu.jl")
else
    init_cuda() = nothing
end

export pull, push, pull_grad, show, affine_pull, affine_push, id, Settings
export registration_operator, vel2mom, vel2mom!, mom2vel, sparsify, greens, kernel, dct, dst, idct, idst, dct!, dst!, idct!, idst!
end

