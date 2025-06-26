import numpy as np
import torch
import copy

def get_input_transform_net(original_net, std_dev=0.1):
    """
    通过向输入添加高斯噪声来应用输入变换防御。
    
    参数:
        original_net: 原始的预测函数。
        std_dev: 高斯噪声的标准差。
        
    返回:
        一个包含了该防御机制的新的预测函数。
    """
    def defended_net(x):
        # x 是一个 PyTorch 张量。我们使用 torch.randn_like 来生成
        # 与 x 具有相同形状、数据类型和设备的噪声。
        noise = torch.randn_like(x) * std_dev
        noisy_x = x + noise 
        return original_net(noisy_x)
    return defended_net

def get_perturbed_model(original_torch_model, std_dev=0.01):
    """
    通过向模型的权重添加噪声来应用模型参数扰动防御。
    这将创建一个模型的扰动副本。

    参数:
        original_torch_model: 原始的PyTorch模型。
        std_dev: 添加到权重上的高斯噪声的标准差。

    返回:
        一个新的、被扰动过的PyTorch模型。
    """
    perturbed_model = copy.deepcopy(original_torch_model)
    with torch.no_grad():
        for param in perturbed_model.parameters():
            noise = torch.randn_like(param) * std_dev
            param.add_(noise)
    return perturbed_model

def get_output_perturbation_net(original_net, num_classes=10, flip_probability=0.1):
    """
    通过随机翻转标签来应用输出扰动防御。
    攻击流程使用的是输出的 argmax，因此我们通过返回一个可能被翻转标签的
    one-hot 编码向量来模拟标签翻转。

    参数:
        original_net: 原始的预测函数。
        num_classes: 输出类别的数量。
        flip_probability: 对任何给定输入翻转标签的概率。

    返回:
        一个包含了该防御机制的新的预测函数。
    """
    def defended_net(x):
        original_output = original_net(x)
        original_labels = np.argmax(original_output, axis=1)
        
        perturbed_labels = original_labels.copy()
        
        for i in range(len(perturbed_labels)):
            if np.random.rand() < flip_probability:
                # 翻转到一个不同的随机类别
                current_label = perturbed_labels[i]
                new_label = np.random.randint(num_classes)
                while new_label == current_label:
                    new_label = np.random.randint(num_classes)
                perturbed_labels[i] = new_label

        # 基于被扰动的标签创建一个 one-hot 编码的输出
        perturbed_output = np.zeros_like(original_output)
        perturbed_output[np.arange(len(perturbed_labels)), perturbed_labels] = 1.0
        
        return perturbed_output 