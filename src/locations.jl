# Doesn't work with ptx files named according to MATLAB's conventions.
"""
   getext()

Get a suitable directory name extension for ptx files etc.
"""
function getext()
    if Sys.iswindows()
        os = "w"
    elseif Sys.islinux()
        os = "a"
    elseif Sys.isapple()
        os = "maci"
    else
        error("not yet sure what to do for this OS")
    end

    word = Sys.WORD_SIZE
    if word==64
        os = os * "64"
    elseif word==32
        od = os * "32"
    else
        error("not yet sure what to do for this word length")
    end
    return os
end


"""
    libdir()

Return location of lib files.
"""
function libdir()
    return joinpath(basedir(), "lib" * getext())
end

function libfile(nam::String)
    return joinpath(libdir(), nam * "." * Libdl.dlext)
end

"""
    ptxdir()

Return location of ptx files.
"""
function ptxdir()
    return joinpath(basedir(), "ptx" * getext())
end

using LazyArtifacts

function basedir()
    if false
        ## Use the following for determining SHA values for Artifacts.toml
        #using Tar, Inflate, SHA
        #filename = "pp_lib.tar.gz";
        #println("sha256 = \"", bytes2hex(open(sha256, filename)), "\"");
        #println("    git-tree-sha1 = \"", Tar.tree_hash(IOBuffer(inflate_gzip(filename))), "\"");

        # This is the path to the Artifacts.toml we will manipulate
        artifact_toml = find_artifacts_toml(@__FILE__)

        # Query the `Artifacts.toml` file for the hash bound to the name "pp_lib"
        # (returns `nothing` if no such binding exists)
        pp_lib_hash = artifact_hash("pp_lib", artifact_toml)
        pp_lib      = artifact_path(pp_lib_hash)
        lib         = artifact"pp_lib"
        return joinpath(lib, "lib")
    else
        return joinpath(@__DIR__, "..", "artifacts", "lib")
    end
end

