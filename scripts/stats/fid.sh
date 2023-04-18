torchrun --standalone --nproc_per_node=1 fid.py calc --images=datasets/ffhq_96_to_128 \
    --ref=datasets/ffhq_128_ref_fid.npz