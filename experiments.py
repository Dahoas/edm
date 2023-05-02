#torchrun --standalone --nnodes=1 --nproc_per_node=8 generate.py --outdir=ffhq_large_def_out --seeds=0-63 --batch=64 \
#    --network ffhq-training-runs/00057-ffhq-128x128_ffhq-96x96_ffhq-64x64_ffhq-48x48_ffhq-32x32-uncond-ddpmpp-edm-gpus8-batch256-fp32/network-snapshot-023206.pkl \
#    --img_resolution 34

import torch
import numpy as np
def bilinear_interpolation(images, new_height, new_width):
    from scipy.interpolate import interp2d
    
    num, c, h, w , d = images.shape
    x = np.linspace(0,1,num=h)
    y = np.linspace(0,1,num=w)

    x2 = np.linspace(0,1,num=new_height)
    y2 = np.linspace(0,1,num=new_width)

    upsampled_images = torch.zeros((num,c,new_height,new_width, d))

    for i in range(num):
        image = images[i]
        for j in range(c):
            cur = image[j]
            for k in range(d):
                curr = cur[:,:,k]
                f = interp2d(y,x,curr.to('cpu').numpy(),kind='linear')

                upsampled_images[i,j,:,:,k] = torch.tensor(f())
    return upsampled_images


modes1 = 16
modes2 = 9
in_channels = 1
out_channels = 1
w1 = torch.rand(in_channels, out_channels, modes1, modes2, 2)

modes1*=2
modes2*=2
print(w1[0,0,:,:,0])
w1 = bilinear_interpolation(w1, modes1,modes2)
print(w1[0,0,:,:,0])
print("Changed the weights")
