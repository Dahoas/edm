import os


if __name__ == "__main__":
    model_path = "lsun_church_training-runs/uno.pkl/network-snapshot-008781.pkl"
    resolutions = [32, 64, 96, 128, 160]
    out_folder = "ffhq_{}_fid_samples"
    ref_path = "datasets/lsun_church_{}.npz"
    for res in resolutions: 
        os.system("bash scripts/infer/fid_sampler.sh {} {} {}".format(model_path, res, out_folder.format(res)))
        print("Computing FID at res {}...".format(res))
        os.system("bash scripts/stats/fid.sh {} {}".format(out_folder.format(res), ref_path.format(res)))
        os.system("rm -r {}".format(out_folder.format(res)))