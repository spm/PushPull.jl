

using CUDA
"""
    lbessi(nu::Real, z::CuArray{Float32})::CuArray{Float32}

Compute `log.(besseli.(nu,z))` on GPU.

"""
function lbessi(nu::Real, z::CuArray{Float32})::CuArray{Float32}
    mod     = CuModuleFile(joinpath(ptxdir(), "lbessi.ptx"));
    fun     = CuFunction(mod, "_Z14lbessi_elementPffPKfy")

    threads = launch_configuration(fun; max_threads=length(z)).threads
    blocks  = Int32(ceil(length(z)/threads))

    od      = CUDA.zeros(Float32, size(z))
    cudacall(fun, (CuPtr{Cfloat},Cfloat,CuPtr{Cfloat},Csize_t),
                  pointer(od),Float32(nu),pointer(z),length(z);
                  threads=threads, blocks=blocks)
    return od
end

