import hydra
from omegaconf import DictConfig
import lightning as lit
from net.base import Net
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import wandb
import torch
from lightning.pytorch.loggers import WandbLogger
from dataset.ds import Dataset

import lightning.pytorch.callbacks as cb

from omegaconf import OmegaConf

import os
import importlib


@hydra.main(
    config_path="/home/softdream/Programming/gits/MNIST-FDS/cfg",
    config_name="base",
    version_base="1.3",
)
def main(cfg: DictConfig):
    print(cfg)
    exit(1)
    lit.seed_everything(cfg.seed)
    wandb_logger = WandbLogger(**cfg.wandb)

    model = Net(cfg.net)
    module = importlib.import_module(cfg.dataset["import"])
    dataset: Dataset = getattr(module, cfg.dataset["name"])(**cfg.dataset["params"])
    train_loader, val_loader, test_loader = dataset.division()
    trainer = lit.Trainer(
        logger=wandb_logger,
        callbacks=[
            cb.EarlyStopping(
                monitor="val_acc", patience=3, verbose=True, mode="max", min_delta=1e-2
            )
        ],
        **cfg.trainer,
    )

    print("Training...")
    trainer.fit(model, train_loader, val_loader)

    print("Testing...")
    trainer.test(model, test_loader)

    hyperparams_dict = OmegaConf.to_container(cfg, resolve=True)
    hyperparams_dict["info"] = {  # type: ignore
        "num_params": get_num_params(model),
    }

    torch.save(model, "weights.pt")
    wandb_logger.log_hyperparams(hyperparams_dict)  # type: ignore


def get_num_params(module):
    """
    Returns the number of parameters in a Lightning module.

    Args:
        module (lightning.pytorch.LightningModule): The Lightning module to get the number of parameters for.

    Returns:
        int: The number of parameters in the module.
    """
    total_params = sum(p.numel() for p in module.parameters())
    return total_params


import convert

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "convert":
            to_convert_path = sys.argv[2]
            convert.build_hydra_configs(to_convert_path, output_dir="cfg2")
            exit(0)
    os.environ["HYDRA_FULL_ERROR"] = "1"
    main()
