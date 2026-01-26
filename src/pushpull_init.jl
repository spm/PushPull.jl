
function __init__()

    global pplib = Libdl.dlopen(libfile("pushpull"))
    global oplib = Libdl.dlopen(libfile("sparse_operator"))
    global tvlib = Libdl.dlopen(libfile("TVdenoise3d"))

    init_cuda()

    return nothing
end

