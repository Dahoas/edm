torchrun --standalone --nnodes=1 --nproc_per_node=8 generate.py --outdir=samples/ffhq_small_model_dual_large_out --seeds=0-63 --batch=8 \
    --network ffhq-training-runs/00080-ffhq_96_to_128-uncond-ddpmpp-edm-gpus8-batch256-fp32/network-snapshot-012544.pkl \
    --img_resolution 128