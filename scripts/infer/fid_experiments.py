import os


if __name__ == "__main__":
    model_path = "lsun-training-runs/00003-lsun_church_96x96-uncond-ddpmpp-edm-gpus8-batch256-fp16/network-snapshot-056448.pkl"
    resolutions = [32,64,96, 128, 160]
    out_folder = "lsun_{}_fid_samples"
    ref_path = "datasets/lsun_church_{}.npz"
    for res in resolutions: 
        os.system("bash scripts/infer/fid_sampler.sh {} {} {}".format(model_path, res, out_folder.format(res)))
        print("Computing FID at res {}...".format(res))
        os.system("bash scripts/stats/fid.sh {} {}".format(out_folder.format(res), ref_path.format(res)))
        os.system("rm -r {}".format(out_folder.format(res)))