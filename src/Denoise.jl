module Denoise
using CUDA
include("denoise3d_gpu.jl")
include("denoise3d_cpu.jl")
export TVdenoise, TVdenoise!
end

