torchrun --standalone --nnodes=1 --nproc_per_node=8 generate.py --outdir=ffhq_large_dual_out --seeds=0-63 --batch=8 \
    --network ffhq-training-runs/00063-ffhq-160x160_ffhq-144x144_ffhq-128x128_ffhq-112x112_ffhq-96x96-uncond-ddpmpp-edm-gpus8-batch256-fp32/network-snapshot-011289.pkl \
    --img_resolution 160