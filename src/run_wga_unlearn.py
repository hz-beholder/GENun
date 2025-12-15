#!/usr/bin/env python3
import os
import torch

from data_tool import DataLoaderTool, DataStore, construct_data
from model_bases import DeepModels
from unlearners import UnlearnWeightedGradientAscent
from utils import seed_everything


def main(
    pretrained_ckpt: str,
    out_dir: str = "/mnt/wanghan/outs/cifar10_-1/cifar10_resnet18/WGA",
    batch_size: int = 256,
    epochs: int = 10,
    beta: float = 5.0,
    forget_size: int = 5000,
    valid_size: int = 5000,
    seed: int = 1,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_everything(seed)

    # 确保输出目录及日志目录存在
    os.makedirs(out_dir, exist_ok=True)
    logs_dir = os.path.join(out_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    robun_log_dir = os.path.join(out_dir, "robun_log")
    os.makedirs(robun_log_dir, exist_ok=True)

    # 1) 准备 CIFAR-10 数据，随机抽取 forget_size 个样本为遗忘集
    train_ds, test_ds = DataLoaderTool.load_dataset(
        dataset_name="cifar10",
        use_default_transform=True,
        augment=True,
    )
    # 使用 construct_data 进行切分（随机 forget_size 个样本作为 forget，valid_size 为验证集大小）
    data, _, _ = construct_data(
        train_data=train_ds,
        test_data=test_ds,
        train_size=-1,
        valid_size=valid_size,
        forget_size=forget_size,
        forget_classes=None,
    )

    # 2) 构建 DeepModels 并加载预训练权重
    feature_dim, num_classes = DataStore.get_dataset_info("cifar10")
    model = DeepModels(
        model_name="resnet18",
        feature_dimension=feature_dim,
        num_classes=num_classes,
        log_path=logs_dir,
        logname=f"cifar10_resnet18_wga_seed{seed}",
        pretrained=True,
    )
    # 设置训练超参
    model.parameter_config(
        device=device,
        name="cifar10_resnet18",
        num_classes=num_classes,
        procedure="WGA",
        lossfn="ce",
        epochs=epochs,
        patience=max(5, epochs // 2),
        optim="adam",
        maxlr=2e-4,
        minlr=1e-4,
        weight_decay=0.0,
        scheduler="none",
        batch_size=batch_size,
        loss_sign=1.0,  
        regularization="none",
        dynamic_regular=False,
    )

    # 加载预训练模型
    assert os.path.isfile(pretrained_ckpt), f"Checkpoint not found: {pretrained_ckpt}"
    model.load(pretrained_ckpt)

    # 3) 启动 WGA 遗忘
    unlearner = UnlearnWeightedGradientAscent(
        logpath=robun_log_dir,
        logname="wga_unlearn",
    )

    # ckpt 前缀（内部会追加 'WGA' 作为名称后缀，并保存训练过程中的检查点）
    ckpt_prefix = os.path.join(out_dir, "checkpoints", "cifar10_resnet18_wga")
    os.makedirs(os.path.dirname(ckpt_prefix), exist_ok=True)

    model_after = unlearner.model_unlearn(
        model=model,
        data=data,
        batch_size=batch_size,
        device=device,
        ckpt_path=ckpt_prefix,
        beta=beta,
    )

    # 4) 保存最终的遗忘模型
    final_out = os.path.join(out_dir, "cifar10_resnet18_wga_final.pth")
    from utils import save_model
    save_model(model_after._model, final_out)
    print(f"[WGA] Done. Saved final model to: {final_out}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="预训练模型路径 .pth/.pt")
    parser.add_argument("--out", type=str, default="/mnt/wanghan/outs/cifar10_-1/cifar10_resnet18/WGA", help="输出目录")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--beta", type=float, default=5.0)
    parser.add_argument("--forget_size", type=int, default=5000)
    parser.add_argument("--valid_size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    main(
        pretrained_ckpt=args.ckpt,
        out_dir=args.out,
        batch_size=args.batch_size,
        epochs=args.epochs,
        beta=args.beta,
        forget_size=args.forget_size,
        valid_size=args.valid_size,
        seed=args.seed,
    )
