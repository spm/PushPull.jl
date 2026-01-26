function init_cuda()
    if any(k->k.name == "CUDA", keys(Base.loaded_modules)) && CUDA.functional()
        global ppmod = CuModuleFile(joinpath(ptxdir(), "pushpull.ptx"))
        global opmod = CuModuleFile(joinpath(ptxdir(), "sparse_operator.ptx"))
        global tvmod = CuModuleFile(joinpath(ptxdir(), "TVdenoise3d.ptx"))
    end
end

