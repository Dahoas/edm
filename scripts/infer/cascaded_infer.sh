torchrun --standalone --nnodes=1 --nproc_per_node=8 generate.py --outdir=ffhq_large_def_out --seeds=0-10 --batch=64 \
    --network /mnt/nvme/home/alex/repos/diffusion/edm/ffhq-training-runs/00067-ffhq-96x96-uncond-ddpmpp-edm-gpus8-batch256-fp32/save/network-snapshot-050678.pkl \
    --img_resolution 96 \
    --cascaded_diffusion_method=random_sample