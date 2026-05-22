module PushPullCUDAExt

try
    print("\n ### Using PushPullCUDAExt ### \n")
    Base.require(Main, :CUDA)
    using CUDA
    using PushPull

    include("pushpull_gpu.jl")
    include("operator_sparse_gpu.jl")
    include("lbessi_gpu.jl")
    include("denoise3d_gpu.jl")

catch
    @warn """Package CUDA not found in current path.
    - Run `import Pkg; Pkg.add(\"CUDA\")` to install it, then restart julia.
    """
end

end

