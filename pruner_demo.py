import warnings

import torch.utils
warnings.filterwarnings('ignore')
import sys, os
sys.path.append(os.path.abspath("../"))

import torch
import torch.nn as nn
import torch_pruning as tp
from model import yolov8n_pvdet
from general import forced_load

class MySlimmingPruner(tp.pruner.MetaPruner):
    def regularize(self, model, reg):
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)) and m.affine==True:
                m.weight.grad.data.add_(reg*torch.sign(m.weight.data)) # Lasso for sparsity


if __name__ == '__main__':
    # 定义模型并设置为评估模式
    model = yolov8n_pvdet(2).eval()
    forced_load(model, "res/pvdetection_07fe2b.pt")
    model = model.backbone

    # 随机生成一个输入张量
    example_inputs = torch.randn(1, 3, 352, 640)

    # 0. importance criterion 
    imp = tp.importance.MagnitudeImportance(p=2)  # L2 范数
    # MySlimmingImportance()

    # 1. ignore some layers that should not be pruned, e.g., the final classifier layer.
    ignored_layers = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d) and m.bias is not None:
            print(f"m={m}")
            ignored_layers.append(m) # DO NOT prune the final classifier!

    # 2. Pruner initialization
    iterative_steps = 5 # You can prune your model to the target pruning ratio iteratively.
    pruner = MySlimmingPruner(
        model, 
        example_inputs, 
        global_pruning=False, # If False, a uniform pruning ratio will be assigned to different layers.
        importance=imp, # importance criterion for parameter selection
        iterative_steps=iterative_steps, # the number of iterations to achieve target pruning ratio
        pruning_ratio=0.5, # remove 50% channels, ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
        ignored_layers=ignored_layers,
    )

    # # Training
    # for _ in range(100):
    #     pass

    base_macs, base_nparams = tp.utils.count_ops_and_params(model, example_inputs)
    for i in range(iterative_steps):
        pruner.step()
        macs, nparams = tp.utils.count_ops_and_params(model, example_inputs)
        print(
            "  Iter %d/%d, Params: %.2f M => %.2f M"
            % (i+1, iterative_steps, base_nparams / 1e6, nparams / 1e6)
        )
        print(
            "  Iter %d/%d, MACs: %.2f G => %.2f G"
            % (i+1, iterative_steps, base_macs / 1e9, macs / 1e9)
        )
        print("="*16)
        # finetune your model here
        # finetune(model)
        # ...
