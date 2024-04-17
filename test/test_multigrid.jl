using CUDA
using PushPull
pp = PushPull

reg = [1e-6, 1., 10., 2.]
#reg = [0., 10., 0., 0.]
reg = [1f-6, 10., 0., 0*100.]
reg = [0., 0., 10., 10]
#reg = [0.,10.,1.,0.]

vx = [1f0,1f0,1f0]

d = (127,131,61)
#d = (128,128,64)
d = (64,64,64)
d = (64,32,2).+1
d = (128,128,64)
#d = (8,8,8);
nsmo = 4

gx = randn(Float32,d)
gy = randn(Float32,d)
gz = randn(Float32,d)
df = randn(Float32,d)

H = zeros(Float32,(d...,6))
g = zeros(Float32,(d...,3))

g[:,:,:,1] .= df.*gx
g[:,:,:,2] .= df.*gy
g[:,:,:,3] .= df.*gz

H[:,:,:,1] .= gx.*gx
H[:,:,:,2] .= gy.*gy
H[:,:,:,3] .= gz.*gz
H[:,:,:,4] .= gx.*gy
H[:,:,:,5] .= gx.*gz
H[:,:,:,6] .= gy.*gz
c  = Int.(round.((d.+1)./2))
if false
    g .= 0; g[c...,1] = 1;
    H[:,:,:,4:6] .=0; H[:,:,:,1:3] .= 0; H[2:end-1,2:end-1,:,:] .= 0; H[c...,1:3] .= 1;
end
#H.=0; #H[:,:,:,1:3].=1;
#g.=9; g[:,:,:,1].=1

if true
print("reg=", reg, " vx=", vx, "\n")
print("\nCPU V-cycle\n")
HLc = pp.hessian_pyramid(H,vx,reg)
vc  = zero(g)
uc  = pp.HLv(vc, HLc)
print(sum((uc.-g).^2),"\n")
for it=1:8
    pp.vcycle!(vc, g, HLc; nit_pre=nsmo, nit_post=nsmo)
    global uc = pp.HLv(vc, HLc)
    print(sum((uc.-g).^2),"\n")
end
end

if true
print("\nGPU V-cycle\n")
gg  = CuArray(g)
Hg  = CuArray(H)
HLg = pp.hessian_pyramid(Hg,vx,reg)
vg  = zero(gg)
ug  = pp.HLv(vg, HLg)
print(sum((ug.-gg).^2),"\n")
for it=1:8
    pp.vcycle!(vg, gg, HLg; nit_pre=nsmo, nit_post=nsmo)
    global ug = pp.HLv(vg, HLg)
    print(sum((ug.-gg).^2),"\n")
end
end

if true
print("\nF-cycle\n")
vc  = zero(g)
uc  = pp.HLv(vc, HLc)
print(sum((uc.-g).^2),"\n")
for it=1:8
    pp.fcycle!(vc, g, HLc; nit_pre=nsmo, nit_post=nsmo)
    global uc = pp.HLv(vc, HLc)
    print(sum((uc.-g).^2),"\n")
end
end

if false
print("\nRelax\n")
HLc = pp.hessian_pyramid(H,vx,reg)
vc = zero(g)
for it=1:8
    pp.relax!(g, HLc,10,vc)
    global uc = pp.HLv(vc,HLc)
    print(sum((uc.-g).^2),"\n")
end
end

