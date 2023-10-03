# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Evaluate Negative Log Likelihood, adapted from the repo by Yang-Song
    https://github.com/yang-song/score_sde_pytorch/tree/main
"""

import os
import re
import click
import tqdm
import pickle
import numpy as np
import torch
import dnnlib
from torch_utils import distributed as dist

from training import dataset
import likelihood

@click.command()
@click.option('--network', 'network_pkl',  help='Network pickle filename', metavar='PATH|URL',                      type=str, required=True)
@click.option('--image_path', 'image_path',help='Path to dataset to evaluate nll in', metavar='PATH',               type=str, required=True)

@click.option('--sigma_min', 'sigma_min',  help='Lowest noise level  [default: varies]', metavar='FLOAT',           type=click.FloatRange(min=0, min_open=True))
@click.option('--sigma_max', 'sigma_max',  help='Highest noise level  [default: varies]', metavar='FLOAT',          type=click.FloatRange(min=0, min_open=True))

def main(network_pkl, image_path, sigma_min=0.002, sigma_max=80.,max_batch_size=64,
    num_workers=3, prefetch_factor=2,device=torch.device('cuda')):
    """Generate random images using the techniques described in the paper
    "Elucidating the Design Space of Diffusion-Based Generative Models".

    Examples:

    \b
    # Generate 64 images and save them as out/*.png
    python generate.py --outdir=out --seeds=0-63 --batch=64 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl

    \b
    # Generate 1024 images using 2 GPUs
    torchrun --standalone --nproc_per_node=2 generate.py --outdir=out --seeds=0-999 --batch=64 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl
    """
    dist.init()

    # Load network.
    dist.print0(f'Loading network from "{network_pkl}"...')
    with dnnlib.util.open_url(network_pkl, verbose=(dist.get_rank() == 0)) as f:
        net = pickle.load(f)['ema'].to(device)

    # List images.
    dist.print0(f'Loading images from "{image_path}"...')
    dataset_obj = dataset.ImageFolderDataset(path=image_path)
    print(len(dataset_obj))

    # Divide images into batches.
    num_batches = ((len(dataset_obj) - 1) // (max_batch_size * dist.get_world_size()) + 1) * dist.get_world_size()
    all_batches = torch.arange(len(dataset_obj)).tensor_split(num_batches)
    rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]
    data_loader = torch.utils.data.DataLoader(dataset_obj, batch_sampler=rank_batches, num_workers=num_workers, prefetch_factor=prefetch_factor)

    # Create data loaders for likelihood evaluation. Only evaluate on uniformly dequantized data
    # train_ds_bpd, eval_ds_bpd, _ = datasets.get_dataset(config,
    #                                                     uniform_dequantization=True, evaluation=True)

    # Go over the dataset 5 times when computing likelihood on the test dataset
    # ds_bpd = eval_ds_bpd
    ds_bpd = data_loader
    bpd_num_repeats = 5

    # Build the likelihood function 
    likelihood_fn = likelihood.get_likelihood_fn(sigma_min,sigma_max)


    # Compute log-likelihoods (bits/dim) 
    bpds = []
    for repeat in range(bpd_num_repeats):
        print(f"Evaluating NLL for the {repeat} time")
        bpd_iter = iter(ds_bpd)  # pytype: disable=wrong-arg-types
        for batch_id in tqdm.tqdm(range(len(ds_bpd))):
            print(batch_id)
            batch = next(bpd_iter)
            # eval_batch = torch.from_numpy(batch['image']._numpy()).to(device).float()
            eval_batch = batch[0].to(device).float()
            # eval_batch = eval_batch.permute(0, 3, 1, 2)
            # eval_batch = scaler(eval_batch)
            bpd = likelihood_fn(net, eval_batch)[0]
            bpd = bpd.detach().cpu().numpy().reshape(-1)
            bpds.extend(bpd)
            print("ckpt: %d, repeat: %d, batch: %d, mean bpd: %6f" % (repeat, batch_id, np.mean(np.asarray(bpds))))
            bpd_round_id = batch_id + len(ds_bpd) * repeat
            # # Save bits/dim to disk or Google Cloud Storage
            # with tf.io.gfile.GFile(os.path.join(eval_dir,
            #                                     f"{config.eval.bpd_dataset}_ckpt_{ckpt}_bpd_{bpd_round_id}.npz"),
            #                         "wb") as fout:
            #     io_buffer = io.BytesIO()
            #     np.savez_compressed(io_buffer, bpd)
            #     fout.write(io_buffer.getvalue())


    torch.distributed.barrier()
    dist.print0('Done.')

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
