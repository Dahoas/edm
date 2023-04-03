torchrun --standalone --nnodes=1 --nproc_per_node=8 experiments.py --outdir=ffhq_cascaded_generation --seeds=0-63 --batch=64 \
    --network /mnt/nvme/home/alex/repos/diffusion/edm/ffhq-training-runs/00066-ffhq-96x96_ffhq-80x80_ffhq-64x64_ffhq-48x48_ffhq-32x32-uncond-ddpmpp-edm-gpus8-batch256-fp32/network-snapshot-055193.pkl \
    --img_resolution 32z