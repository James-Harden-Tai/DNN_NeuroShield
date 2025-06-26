import argparse
import os
import sys
import numpy as np
import torch
import shutil
from multiprocessing import Pool

# 允许脚本从父目录导入模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from signature_recovery import utils, find_duals, cluster_dual_points, recover_weights
from signature_recovery.defenses import (
    get_input_transform_net,
    get_perturbed_model,
    get_output_perturbation_net,
)

def run_pipeline(defense_name, num_runs=1, noise_std=0.1, flip_prob=0.1, target_layer=0, num_clusters=64):
    """
    运行完整的攻击流程，并应用指定的防御策略。
    :param defense_name: 要应用的防御策略名称 ('input', 'model', 'output', or 'none')。
    :param num_runs: 查找对偶点的运行次数。
    :param noise_std: 用于输入/模型防御的噪声标准差。
    :param flip_prob: 用于输出防御的标签翻转概率。
    :param target_layer: 要攻击的目标层ID。
    :param num_clusters: 要生成的最大聚类数。
    """
    base_dir = f"{defense_name}_exp"
    logs_dir = f"{defense_name}_logs"
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    if os.path.exists(logs_dir):
        shutil.rmtree(logs_dir)
    os.makedirs(base_dir)
    os.makedirs(logs_dir)

    print(f"正在使用 '{defense_name}' 防御策略运行攻击流程...")

    # 1. 根据防御策略选择或构建 bmodel 函数
    bmodel_fn = utils.bmodel
    if defense_name == "input":
        # 如果是输入转换防御，我们需要创建一个包裹了原始 bmodel 的新函数
        bmodel_fn = get_input_transform_net(utils.bmodel, std_dev=noise_std)
    elif defense_name == "model":
        # 模型扰动会返回一个新的、被扰动过的模型。我们需要用这个新模型来更新 utils 中的 bmodel。
        perturbed_model = get_perturbed_model(utils.cheat_net_cuda, std_dev=noise_std)
        utils.cheat_net_cuda = perturbed_model
        # bmodel_fn 保持不变，但它现在会在内部使用被扰动过的模型
        bmodel_fn = utils.bmodel 
    elif defense_name == "output":
        # 输出扰动防御同样包裹原始的 bmodel
        bmodel_fn = get_output_perturbation_net(utils.bmodel, flip_probability=flip_prob)

    # 2. 运行 find_duals.py
    # 将选择的（可能带有防御的）bmodel_fn 传递给 main 函数
    print(f"[{defense_name}] 阶段 1: 查找对偶点...")
    find_duals_args = argparse.Namespace(
        output=base_dir, cpus=os.cpu_count(), runs=num_runs
    )
    # 我们不再需要并行运行，因为 find_duals 的并行化已在内部处理
    find_duals.main(args=find_duals_args, bmodel_fn=bmodel_fn)

    # 3. 运行 cluster_dual_points.py
    print(f"[{defense_name}] 阶段 2: 聚类对偶点...")
    cluster_args = argparse.Namespace(input=base_dir, output=base_dir, num_clusters=num_clusters)
    cluster_dual_points.main(cluster_args)

    # 4. 运行 recover_weights.py 并记录日志
    log_file_path = os.path.join(logs_dir, f"recover_weights_layer_{target_layer}.log")
    print(f"[{defense_name}] 阶段 3: 恢复权重... (日志将保存到 {log_file_path})")
    
    recover_args = argparse.Namespace(
        input=base_dir,
        output=base_dir,
        layer=target_layer,
    )
    
    original_stdout = sys.stdout
    with open(log_file_path, 'w') as log_file:
        sys.stdout = log_file
        try:
            recover_weights.main(recover_args)
        finally:
            sys.stdout = original_stdout

    print(f"'{defense_name}' 防御策略下的攻击流程已完成。实验数据保存在 '{base_dir}'，详细日志保存在 '{logs_dir}'。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="使用不同的防御策略运行攻击流程。"
    )
    parser.add_argument(
        "--defense",
        type=str,
        required=True,
        choices=["input", "model", "output", "none"],
        help="要应用的防御策略。",
    )
    parser.add_argument(
        "--num_runs", type=int, default=1, help="查找对偶点的运行次数。"
    )
    parser.add_argument(
        "--noise_std", type=float, default=0.1, help="用于输入防御的噪声标准差。"
    )
    parser.add_argument(
        "--flip_prob", type=float, default=0.25, help="用于输出防御的标签翻转概率。"
    )
    parser.add_argument(
        "--layer", type=int, default=0, help="要攻击的目标层。"
    )
    parser.add_argument(
        "--num_clusters", type=int, default=64, help="要生成的最大聚类数（默认为64）。"
    )
    args = parser.parse_args()

    # 运行完整的攻击流程
    run_pipeline(args.defense, args.num_runs, args.noise_std, args.flip_prob, args.layer, args.num_clusters)

    print("\n*** 防御实验成功完成！ ***") 