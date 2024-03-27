using CUDA
using PushPull
pp = PushPull

reg = [1e-6, 1., 10., 2.]
#reg = [0., 10., 0., 0.]
reg = [0., 0., 10., 0.]

d = (127,131,61)
#d = (128,128,64)
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


HLc = pp.hessian_pyramid(H,[1f0,1f0,1f0],reg)
vc  = zero(g)
uc  = pp.HLv(vc, HLc)
print(sum((uc.-g).^2),"\n")
for it=1:16
    pp.vcycle!(vc, g, HLc; nit_pre=4, nit_post=4)
    global uc = pp.HLv(vc, HLc)
    print(sum((uc.-g).^2),"\n")
end

gg  = CuArray(g)
Hg  = CuArray(H)
HLg = pp.hessian_pyramid(Hg,[1f0,1f0,1f0],reg)
vg  = zero(gg)
ug  = pp.HLv(vg, HLg)
print(sum((ug.-gg).^2),"\n")
for it=1:16
    pp.vcycle!(vg, gg, HLg; nit_pre=4, nit_post=4)
    global ug = pp.HLv(vg, HLg)
    print(sum((ug.-gg).^2),"\n")
end

