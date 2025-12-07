from __future__ import annotations

import argparse
import importlib
from typing import TYPE_CHECKING

import numpy as np
import torch

from utmosv2._settings import configure_defaults, configure_inference_args
from utmosv2.runner import run_inference
from utmosv2.utils import (
    get_dataloader,
    get_dataset,
    get_inference_data,
    get_model,
    make_submission_file,
    print_metrics,
    save_preds,
    save_test_preds,
    show_inference_data,
)

if TYPE_CHECKING:
    from utmosv2._settings._config import Config


def main(cfg: Config) -> None:
    data = get_inference_data(cfg)
    show_inference_data(data)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg.print_config = True  # type: ignore

    test_preds = np.zeros(data.shape[0])
    if cfg.reproduce:
        test_metrics: dict[str, float] = {}

    for fold in range(cfg.num_folds):
        if 0 <= cfg.inference.fold < cfg.num_folds and fold != cfg.inference.fold:
            continue

        cfg.now_fold = fold  # type: ignore

        model = get_model(cfg, device)

        cfg.print_config = False  # type: ignore
        print(f"+*+*[[Fold {fold + 1}/{cfg.num_folds}]]" + "+*" * 30)

        for cycle in range(cfg.inference.num_tta):
            test_dataset = get_dataset(cfg, data, "test")
            test_dataloader = get_dataloader(cfg, test_dataset, "test")
            test_preds_tta, test_metrics_tta = run_inference(
                cfg, model, test_dataloader, cycle, data, device
            )
            test_preds += test_preds_tta
            if cfg.reproduce:
                assert test_metrics_tta is not None
                for k, v in test_metrics_tta.items():
                    test_metrics[k] = test_metrics.get(k, 0) + v

    fold_cnt = 1 if 0 <= cfg.inference.fold < cfg.num_folds else cfg.num_folds
    print(f"Average of {fold_cnt} folds")
    test_preds /= fold_cnt * cfg.inference.num_tta
    if cfg.reproduce:
        test_metrics = {
            k: v / fold_cnt / cfg.inference.num_tta for k, v in test_metrics.items()
        }
        print_metrics(test_metrics)
        save_test_preds(cfg, data, test_preds, test_metrics)
        make_submission_file(cfg, data, test_preds)
    else:
        save_preds(cfg, data, test_preds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=str, default="fusion_stage3", help="config file name"
    )
    parser.add_argument("-f", "--fold", type=int, default=0, help="fold number")
    parser.add_argument(
        "-s", "--seed", type=int, default=42, help="random seed for split"
    )
    parser.add_argument("-d", "--input_dir", type=str, help="data path")
    parser.add_argument("-p", "--input_path", type=str, help="data path")
    parser.add_argument("-o", "--out_path", type=str, help="output path")
    parser.add_argument(
        "-n",
        "--num_workers",
        type=int,
        default=4,
        help="number of workers for dataloader",
    )
    parser.add_argument(
        "-t",
        "--val_list_path",
        type=str,
        help="test data path",
    )
    parser.add_argument(
        "-w", "--weight", type=str, default=None, help="path to the weight file to load"
    )
    parser.add_argument(
        "-pd",
        "--predict_dataset",
        type=str,
        default="sarulab",
        help="predict dataset",
    )
    parser.add_argument(
        "-nr",
        "--num_repetitions",
        type=int,
        default=1,
        help="number of repetitions for prediction",
    )
    parser.add_argument(
        "-e",
        "--reproduce",
        action="store_true",
        help="Run the experiment as described in the paper, including all necessary steps for reproducibility.",
    )
    parser.add_argument(
        "-fi",
        "--final",
        action="store_true",
        help="final submission",
    )
    args = parser.parse_args()

    if args.input_dir is None and args.input_path is None:
        raise ValueError(
            "Either input_dir or input_path must be provided when you use your own data."
        )
    if args.input_dir is not None and args.input_path is not None:
        raise ValueError(
            "Only one of input_dir or input_path must be provided when you use your own data."
        )

    cfg = importlib.import_module("utmosv2.config." + args.config)
    configure_inference_args(cfg, args)
    configure_defaults(cfg)

    main(cfg)
