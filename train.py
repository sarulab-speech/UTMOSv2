import argparse
import importlib
import os

import numpy as np
import torch
from utmosv2._settings._config import Config
import wandb
from dotenv import load_dotenv

from utmosv2._settings import configure_args, configure_defaults
from utmosv2.runner import run_train
from utmosv2.utils import (
    get_dataloader,
    get_dataset,
    get_loss,
    get_metrics,
    get_model,
    get_optimizer,
    get_scheduler,
    get_train_data,
    save_oof_preds,
    split_data,
)


def main(cfg: Config) -> None:
    data = get_train_data(cfg)
    print(data.head())
    oof_preds = np.zeros(data.shape[0])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cfg.print_config = True  # type: ignore

    for fold, (train_idx, val_idx) in enumerate(split_data(cfg, data)):
        if 0 <= cfg.fold < cfg.num_folds and fold != cfg.fold:
            continue

        cfg.now_fold = fold  # type: ignore

        train_data = data.iloc[train_idx]
        val_data = data.iloc[val_idx]

        train_dataset = get_dataset(cfg, train_data, "train")
        val_dataset = get_dataset(cfg, val_data, "valid")

        train_dataloader = get_dataloader(cfg, train_dataset, "train")
        val_dataloader = get_dataloader(cfg, val_dataset, "valid")

        model = get_model(cfg, device)
        criterions = get_loss(cfg)
        metrics = get_metrics()
        optimizer = get_optimizer(cfg, model)
        scheduler = get_scheduler(
            cfg, optimizer, len(train_dataloader) * cfg.run.num_epochs
        )

        cfg.print_config = False  # type: ignore
        print(f"+*+*[[Fold {fold + 1}/{cfg.num_folds}]]" + "+*" * 30)
        if cfg.wandb:
            wandb.init(
                project="voice-mos-challenge-2024",
                name=cfg.config_name,
                config={
                    "fold": fold,
                    "seed": cfg.split.seed,
                },
            )

        run_train(
            cfg,
            model,
            train_dataloader,
            val_dataloader,
            val_data,
            oof_preds,
            fold,
            criterions,
            metrics,
            optimizer,
            scheduler,
            device,
        )
        if cfg.wandb:
            wandb.finish()

    save_oof_preds(cfg, data, oof_preds, cfg.fold)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="config file name"
    )
    parser.add_argument("-f", "--fold", type=int, default=-1, help="fold number")
    parser.add_argument(
        "-s", "--seed", type=int, default=42, help="random seed for split"
    )
    parser.add_argument(
        "-i", "--input_dir", type=str, default="data/main/DATA", help="data path"
    )
    parser.add_argument(
        "-dc", "--data_config", type=str, help="path to the data config file"
    )
    parser.add_argument(
        "-n",
        "--num_workers",
        type=int,
        default=4,
        help="number of workers for dataloader",
    )
    parser.add_argument(
        "-w", "--weight", type=str, help="path to the weight file to load"
    )
    parser.add_argument(
        "-e",
        "--reproduce",
        action="store_true",
        help="Run the experiment as described in the paper, including all necessary steps for reproducibility.",
    )
    parser.add_argument(
        "-wb", "--wandb", action="store_true", help="Use wandb for logging"
    )
    args = parser.parse_args()

    if args.reproduce is None and args.data_config is None:
        raise ValueError("Either --reproduce or --data_config must be specified")

    cfg = importlib.import_module("utmosv2.config." + args.config)
    configure_args(cfg, args)
    configure_defaults(cfg)

    load_dotenv()
    if cfg.wandb:
        wandb.login(key=os.getenv("WANDB_API_KEY"))

    main(cfg)
