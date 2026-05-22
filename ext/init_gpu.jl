# unused, but kept for reference
function init_cuda()
    if any(k->k.name == "CUDA", keys(Base.loaded_modules)) && CUDA.functional()
        global ppmod = CuModuleFile(joinpath(PushPull.ptxdir(), "pushpull.ptx"))
        global opmod = CuModuleFile(joinpath(PushPull.ptxdir(), "sparse_operator.ptx"))
        global tvmod = CuModuleFile(joinpath(PushPull.ptxdir(), "TVdenoise3d.ptx"))
    end
end

