import matplotlib.pyplot as plt
import numpy as np
import json


def load_jsonl(filename):
    data = []
    with open(filename, "r") as f:
        lines = f.readlines()
        for line in lines:
            response = json.loads(line)
            data.append(response)
    return data


if __name__ == "__main__":
    stats_path = "lsun_church_training-runs/00002-____-uncond-ddpmpp-edm-gpus8-batch256-fp32/stats.jsonl"
    stats = load_jsonl(stats_path)
    stats = stats[2:]
    plt.clf()
    high = 0
    high_key = None
    for key in stats[0]:
        if "loss" in key:
            xs = np.arange(len(stats))
            ys = [stat[key]["mean"] for stat in stats]
            plt.plot(xs, ys, "-o", label=key)
            try:
                if high < int(key.split("x")[-1]):
                    high = int(key.split("x")[-1])
                    high_key = key
            except:
                high_key = None
    plt.legend()
    plt.savefig("loss.png")

    if high_key is not None:
        high = str(high)
        plt.clf()
        plt.plot(xs, [stat[high_key]["mean"] for stat in stats], "-o", label=high_key)
        plt.legend()
        plt.savefig("high_loss.png")