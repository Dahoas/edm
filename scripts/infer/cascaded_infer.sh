torchrun --standalone --nnodes=1 --nproc_per_node=8 generate.py --outdir=ffhq_large_def_out/normal/160/ --seeds=0-10 --batch=64 \
    --network /mnt/nvme/home/alex/repos/diffusion/edm/ffhq-training-runs/00073-ffhq-96x96_ffhq-80x80_ffhq-64x64_ffhq-48x48_ffhq-32x32-uncond-ddpmpp-edm-gpus8-batch256-fp32/network-snapshot-107878.pkl \
    --img_resolution 160 --upsample_resolution 160 \
    --cascaded_diffusion_method=random_sample