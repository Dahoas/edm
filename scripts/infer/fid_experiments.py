import os


if __name__ == "__main__":
    model_path = "lsun_church_training-runs/00001-____-uncond-ddpmpp-edm-gpus8-batch256-fp32/network-snapshot-104115.pkl"
    resolutions = [32, 64, 96, 128, 160,196]
    out_folder = "lsun_{}_fid_samples"
    ref_path = "datasets/lsun_church_{}.npz"
    for res in resolutions: 
        if res == 32:
            continue

        os.system("bash scripts/infer/fid_sampler.sh {} {} {}".format(model_path, res, out_folder.format(res)))
        print("Computing FID at res {}...".format(res))
        os.system("bash scripts/stats/fid.sh {} {}".format(out_folder.format(res), ref_path.format(res)))
        os.system("rm -r {}".format(out_folder.format(res)))