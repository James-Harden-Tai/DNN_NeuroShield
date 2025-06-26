import re
import os
import sys
import pickle
from utils import *
from collections import defaultdict
import argparse

from recover_weights import is_consistent, CIFAR10NetPrefix, transfer_weights


def cheat_cluster(layer):
    duals = []
    root = "exp/1/"
    for f in sorted(os.listdir(root)):
        print(f)
        x = pickle.load(open(os.path.join(root, f), "rb"))
        duals.extend(x)

    cheating = defaultdict(list)
    for idx, (left, middle, right) in enumerate(duals):
        if idx % 1000 == 0:
            print(idx, "/", len(duals))
        diff = cheat_neuron_diff_cuda(left, right)
        if len(diff) == 1:
            if diff[0] // DIM == layer:
                cheating[diff[0]].append((left, middle, right))

    pickle.dump(cheating, open("exp/1-cluster-%d.p" % layer, "wb"))


def refine_cluster(maybe, layer, prefix):
    maybe = np.array(maybe)
    points = np.zeros(len(maybe))
    for _ in range(10):
        order = np.arange(len(maybe))
        random.shuffle(order)
        for i in range(0, len(order) - (len(order) % 3), 3):
            ok = is_consistent(
                [maybe[x] for x in order[i : i + 3]], prefix, layer, False
            )
            print("ok?", ok)
            if ok is not None and ok > 1e-5:
                points[order[i : i + 3]] += 1

    maybe = maybe[points < 6]
    return maybe


def cluster_slow(layer):
    prefix = CIFAR10NetPrefix(layer).cuda()
    transfer_weights(cheat_net_cpu, prefix)

    duals = []
    root = "exp/1/"
    for f in sorted(os.listdir(root)):
        print(f)
        x = pickle.load(open(os.path.join(root, f), "rb"))
        duals.extend(x)

    output = {}
    print("LAYER", layer)
    for cluster_id, a in enumerate(duals[:1000]):
        print("idx", cluster_id)
        maybe = [a]
        for j, b in enumerate(duals):
            if j > 1000 and len(maybe) < 3000 / j:
                print("Too low rate; break", j)
                break
            S = is_consistent((a, b), prefix, False)
            # Necessary to tune 1e-5 for the appropriate TPR/FPR tradeoff
            if type(S) == np.float64 and S < 1e-5:
                print("Got close")
                print(
                    S,
                    cheat_neuron_diff_cuda(a[0], a[2]),
                    cheat_neuron_diff_cuda(b[0], b[2]),
                )
                maybe.append(b)
        print("Found set size", len(maybe))

        print("Before refine")
        for i in range(len(maybe)):
            idx = cheat_neuron_diff_cuda(maybe[i][0], maybe[i][2])
            print(idx)

        # OPTIONAL: increase precision, reduce recall
        if len(maybe) > 2:
            maybe = refine_cluster(maybe, layer, prefix)
        else:
            continue

        print("After refine")
        for i in range(len(maybe)):
            idx = cheat_neuron_diff_cuda(maybe[i][0], maybe[i][2])
            print(idx)

        print("WRITING", cluster_id)
        output[cluster_id] = maybe
        pickle.dump(output, open("exp/1-cluster-%d.p" % layer, "wb"))


def main(args=None):
    if not args:
        parser = argparse.ArgumentParser(description="Cluster dual points from a folder.")
        parser.add_argument("input", type=str, help="Folder containing dual point .p files.")
        parser.add_argument("--output", type=str, help="Output folder for the clustered points.")
        parser.add_argument("--num_clusters", type=int, default=None, help="要生成的最大聚类数。如果为None，则保留所有找到的聚类。")
        args = parser.parse_args()

    # The input folder is now from the args object
    folder = args.input
    output_dir = args.output if args.output else folder
    
    # This script is a bit brittle, we have to reconstruct the cheat_cluster logic
    # but with the correct folder paths from the main runner script.
    duals = []
    # root folder is now the argument
    root = folder
    for f in sorted(os.listdir(root)):
        if not f.endswith('.p'):
            continue
        print(f"Loading duals from: {os.path.join(root, f)}")
        with open(os.path.join(root, f), "rb") as file_handle:
            x = pickle.load(file_handle)
            duals.extend(x)

    # We assume layer 0 for the TINY model, and layer 1 for others
    # This is a bit of a magic number, but reflects the project's structure.
    layer = 1
    cheating = defaultdict(list)
    for idx, (left, middle, right) in enumerate(duals):
        if idx % 1000 == 0:
            print(f"Processing dual point {idx}/{len(duals)}")
        diff = cheat_neuron_diff_cuda(left, right)
        if len(diff) == 1:
            # For the TINY model, the layer check might be different.
            # Assuming we are interested in the first hidden layer (layer 1)
            # The dimension (DIM) is 784 for tiny model's input
            # Let's assume the first hidden layer has neuron indices starting from where the input dim ends.
            # The logic `diff[0] // DIM == layer` might need adjustment based on model architecture.
            # For now, let's just cluster by neuron ID.
            cheating[diff[0]].append((left, middle, right))
    
    # 根据用户要求，将聚类数量限制为指定数量
    num_clusters_to_keep = args.num_clusters if hasattr(args, 'num_clusters') else None
    if num_clusters_to_keep is not None and len(cheating) > num_clusters_to_keep:
        print(f"Found {len(cheating)} clusters, limiting to {num_clusters_to_keep}.")
        # To make it deterministic, sort by neuron ID before taking the top N
        sorted_keys = sorted(cheating.keys())
        limited_cheating = {k: cheating[k] for k in sorted_keys[:num_clusters_to_keep]}
else:
        limited_cheating = cheating
        if num_clusters_to_keep is not None:
             print(f"Found {len(cheating)} clusters, which is not more than {num_clusters_to_keep}. Keeping all.")

    # Save the limited dictionary of clusters to a single file.
    output_path = os.path.join(output_dir, "clusters.p")
    with open(output_path, "wb") as file_handle:
        pickle.dump(limited_cheating, file_handle)
    
    print(f"Clustering results for {len(limited_cheating)} clusters saved to: {output_path}")

if __name__ == '__main__':
    # This allows running the script standalone
    main()
