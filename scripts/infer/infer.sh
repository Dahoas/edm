torchrun --standalone --nnodes=1 --nproc_per_node=8 generate.py --outdir=samples/ffhq_small_dual_out --seeds=0-63 --batch=8 \
    --network ffhq-training-runs/00123-ffhq-96x96-to-160x160_ffhq-96x96_ffhq-80x80_ffhq-64x64_ffhq-48x48_ffhq-32x32-uncond-ddpmpp-edm-gpus8-batch256-fp32/network-snapshot-001254.pkl \
    --img_resolution 192