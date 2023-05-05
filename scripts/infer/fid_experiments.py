import os


if __name__ == "__main__":
    model_path = "ffhq-training-runs/00075-ffhq-96x96_ffhq-80x80_ffhq-64x64_ffhq-48x48_ffhq-32x32-uncond-ddpmpp-edm-gpus8-batch256-fp32/network-snapshot-020070.pkl"
    resolutions = [32, 64, 96, 128, 160]
    out_folder = "ffhq_{}_fid_samples"
    ref_path = "datasets/ffhq_{}_ref_fid.npz"
    for res in resolutions: 
        os.system("bash scripts/infer/fid_sampler.sh {} {} {}".format(model_path, res, out_folder.format(res)))
        print("Computing FID at res {}...".format(res))
        os.system("bash scripts/stats/fid.sh {} {}".format(out_folder.format(res), ref_path.format(res)))
        os.system("rm -r {}".format(out_folder.format(res)))