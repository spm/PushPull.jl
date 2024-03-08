
function __init__()

    if CUDA.functional()
        # If there's no GPU, then CuModuleFile will not run so the various globals
        # remain undefined. This should not matter because no data should be
        # transferred to GPU and none of the various GPU functions are called.
        # Instead, users will see the following warning:
        #   ┌ Info: The CUDA function is being called but CUDA.jl is not functional.
        #   └ Defaulting back to the CPU. (No action is required if you want to run on the CPU).

        # Globals used by pushpull_gpu.jl
        global ppmod      = CuModuleFile(joinpath(ptxdir(), "pushpull.ptx"))
        global cuPull     = CuFunction(ppmod, "_Z12pull_elementPfPKfS1_")
        global cuPullGrad = CuFunction(ppmod, "_Z13pullg_elementPfPKfS1_")
        global cuPullHess = CuFunction(ppmod, "_Z13pullh_elementPfPKfS1_")
        global cuPush     = CuFunction(ppmod, "_Z12push_elementPfPKfS1_")
        global cuPushGrad = CuFunction(ppmod, "_Z13pushg_elementPfPKfS1_")
        global cuAffPull  = CuFunction(ppmod, "_Z19affine_pull_elementPfPKf")
        global cuAffPush  = CuFunction(ppmod, "_Z19affine_push_elementPfPKf")

        # Globals used by sparse_operator.jl
        global opmod        = CuModuleFile(joinpath(ptxdir(), "sparse_operator.ptx"))
        global cuVel2mom    = CuFunction(opmod, "_Z15vel2mom_elementPfPKf")
        global cuVel2momPad = CuFunction(opmod, "_Z22vel2mom_padded_elementPfPKf")
        global cuRelax      = CuFunction(opmod, "_Z13relax_elementPfPKfS1_")
        global cuRelaxPad   = CuFunction(opmod, "_Z20relax_padded_elementPfPKfS1_")

    end

    # Global used by pushpull_cpu.jl
    global pplib = Libdl.dlopen(libfile("pushpull"))
    global oplib = Libdl.dlopen(libfile("sparse_operator"))

    return nothing
end

