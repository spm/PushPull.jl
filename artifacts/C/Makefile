include Makefile.var

PTXFILES = pushpull.ptx lbessi.ptx sparse_operator.ptx TVdenoise3d.ptx TVdenoise2d.ptx blip.ptx
LIBFILES = pushpull.$(SOSUF) TVdenoise3d.$(SOSUF) sparse_operator.$(SOSUF) blip.$(SOSUF)
PPLIB    = ../pp_lib.tar.gz

CUDA_VERSION := $(shell $(NVCC) --version 2> /dev/null)
ifndef CUDA_VERSION
	PTXFILES =
	PTXDIR =
endif

all: $(PTXFILES) $(LIBFILES)

install: all $(PTXDIR) $(LIBDIR)
ifdef CUDA_VERSION
	$(COPY) $(PTXFILES) $(PTXDIR)
endif
	$(COPY) $(LIBFILES) $(LIBDIR)

tarfile:
	tar -C .. -cvf - lib | gzip -v > $(PPLIB)
	julia get_sha.jl $(PPLIB)

clean:
	$(DEL) $(PTXFILES) $(LIBFILES)

%.$(SOSUF) : %.c
	$(CC) -O3 -shared -fPIC -Wl,-$(SONAME),$@ -o $@ -lm $<

%.ptx : %.cu
	$(NVCC) -ptx --ptxas-options --verbose $< 

$(PTXDIR) :
	$(MKDIR) $(PTXDIR)

$(LIBDIR) :
	$(MKDIR) $(LIBDIR)
