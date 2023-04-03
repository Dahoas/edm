#torchrun --standalone --nnodes=1 --nproc_per_node=8 generate.py --outdir=ffhq_large_def_out --seeds=0-63 --batch=64 \
#    --network ffhq-training-runs/00057-ffhq-128x128_ffhq-96x96_ffhq-64x64_ffhq-48x48_ffhq-32x32-uncond-ddpmpp-edm-gpus8-batch256-fp32/network-snapshot-023206.pkl \
#    --img_resolution 34

from  generate.py import *

image = generate.edm_sampler

# Random Generator
rnd = StackedRandomGenerator(device, batch_seeds)
latents = rnd.randn([batch_size, net.img_channels, img_resolution, img_resolution], device=device)

# Load network.
dist.print0(f'Loading network from "{network_pkl}"...')
with dnnlib.util.open_url(network_pkl, verbose=(dist.get_rank() == 0)) as f:
    net = pickle.load(f)['ema'].to(device)        

# Sample
sampler_fn = edm_sampler
images = sampler_fn(net, latents, randn_like=rnd.randn_like)

# Save images.
images_np = (images * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
#os.makedirs(outdir, exist_ok=True)
#np.save(os.path.join(outdir, "batch_{}".format(batch_seeds[0])), images_np)
for seed, image_np in zip(batch_seeds, images_np):
    image_dir = os.path.join(outdir, f'{seed-seed%1000:06d}') if subdirs else outdir
    os.makedirs(image_dir, exist_ok=True)
    image_path = os.path.join(image_dir, f'{seed:06d}.png')
    if image_np.shape[2] == 1:
        PIL.Image.fromarray(image_np[:, :, 0], 'L').save(image_path)
    else:
        PIL.Image.fromarray(image_np, 'RGB').save(image_path)
