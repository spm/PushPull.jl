using Tar, Inflate, SHA
filename = "../pp_lib.tar.gz"
if length(ARGS)>=1
    filename = ARGS[1]
end
sha1    = Tar.tree_hash(IOBuffer(inflate_gzip(filename)))
sha2 = bytes2hex(open(sha256, filename))
println("Edit the Artifacts.toml using the following:\n")
println("git-tree-sha1 = \"", sha1, "\"");
println("    sha256 = \"", sha2, "\"");

println("\nor run the following (in Unix):")
artif = "../../Artifacts.toml"
println("\ncp ", artif, " Artifacts.prev ; sed < Artifacts.prev '/git-tree-sha1/s/\"[0-9,a-f].*\"/\"", sha1,"\"/' | sed '/sha256/s/\"[0-9,a-f].*\"/\"",sha2,"\"/' > ", artif, "\n")

