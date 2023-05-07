model_path=$1
res=$2
outdir=$3
torchrun --standalone --nnodes=1 --nproc_per_node=8 generate.py --outdir=$outdir --seeds=0-49999 --batch=128 \
    --network $model_path \
    --img_resolution $res \
    --cascaded_diffusion_method random_sample