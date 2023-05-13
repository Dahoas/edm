torchrun --standalone --nnodes=1 --nproc_per_node=8 generate.py --outdir=lsun_fno_out --seeds=0-10 --batch=8 \
    --network lsun-training-runs/00033-lsun_church_96x96_lsun_church_80x80_lsun_church_64x64_lsun_church_48x48_lsun_church_32x32-uncond-ddpmpp-edm-gpus8-batch256-fp32/network-snapshot-178125.pkl \
    --img_resolution 96