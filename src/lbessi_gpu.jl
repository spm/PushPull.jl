

using CUDA
"""
    lbessi(nu::Real, zd::CuArray{Float32})::CuArray{Float32}

Compute `log(besseli(nu,z))` on GPU.
"""
function lbessi(nu::Real, zd::CuArray{Float32})::CuArray{Float32}
    mod     = CuModuleFile(joinpath(ptxdir(), "lbessi.ptx"));
    fun     = CuFunction(mod, "_Z14lbessi_elementPffPKfy")

    dev     = CUDA.device()
    threads = min(attribute(dev,CUDA.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK),length(zd))
    blocks  = Int32(ceil(length(zd)/threads))

    od = CUDA.zeros(Float32, size(zd))
    cudacall(fun, (CuPtr{Cfloat},Cfloat,CuPtr{Cfloat},Csize_t),
                  pointer(od),Float32(nu),pointer(zd),length(zd);
                  threads=threads, blocks=blocks)
    return od
end

