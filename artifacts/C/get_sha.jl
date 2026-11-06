using Tar, Inflate, SHA
filename = "../pp_lib.tar.gz"
if length(ARGS)>=1
    filename = ARGS[1]
end

# A copy of the last Artifacts.toml file, which is no longer used
art  = """
[pp_lib]
git-tree-sha1 = "8afb0ad82b0404fedd317fe4e7e42b2b27ee1bf1"
lazy = true
    [[pp_lib.download]]
    sha256 = "ad112a0d92e691639252ffa28cc50a09e080cd3488170e4e24fc9ee25feb901a"
    url = "https://raw.githubusercontent.com/spm/PushPull.jl/main/artifacts/pp_lib.tar.gz"
"""

sha1 = Tar.tree_hash(IOBuffer(inflate_gzip(filename)))
sha2 = bytes2hex(open(sha256, filename))
println("Edit the Artifacts.toml using the following:\n")
println("git-tree-sha1 = \"", sha1, "\"");
println("    sha256 = \"", sha2, "\"");

println("\nor run the following (in Unix):")
artif = "../../Artifacts.toml"
println("\ncp ", artif, " Artifacts.prev ; sed < Artifacts.prev '/git-tree-sha1/s/\"[0-9,a-f].*\"/\"", sha1,"\"/' | sed '/sha256/s/\"[0-9,a-f].*\"/\"",sha2,"\"/' > ", artif, "\n")

#println("\ncp ", artif, " Artifacts.prev\nsed < Artifacts.prev '/git-tree-sha1/s/\"[0-9,a-f].*\"/\"", sha1,"\"/' | sed '/sha256/s/\"[0-9,a-f].*\"/\"",sha2,"\"/' | sed '/url/s/pp_lib_*[0-9,a-f]*/pp_lib_", sha2[1:8], "/' > ", artif, "\nmv ", filename, " ../pp_lib_", sha2[1:8], ".tar.gz\n")

