torchrun --standalone --nnodes=1 --nproc_per_node=8 generate.py --outdir=ffhq_cascaded_generation --seeds=0-63 --batch=64 \
    --network https://drive.google.com/file/d/1B4r0ibrgh3EoqyaLVMQxSfCmOibu_9k9/view?usp=sharing \
    --img_resolution 32