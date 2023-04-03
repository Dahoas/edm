torchrun --standalone --nnodes=1 --nproc_per_node=8 generate.py --outdir=ffhq_fid_small_def_single_res_out --seeds=0-49999 --batch=128 \
    --network ffhq-training-runs/00067-ffhq-96x96-uncond-ddpmpp-edm-gpus8-batch256-fp32/network-snapshot-001003.pkl \
    --img_resolution 96