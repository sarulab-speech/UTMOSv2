from __future__ import annotations

import argparse
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias

# NOTE: Python 3.12 introduces the type statement, so once Python 3.11 is dropped,
# it should be updated to use that instead.
Config: TypeAlias = SimpleNamespace | ModuleType


def configure_args(cfg: Config, args: argparse.Namespace) -> None:
    cfg.fold = args.fold  # type: ignore
    cfg.split.seed = args.seed  # type: ignore
    cfg.config_name = args.config  # type: ignore
    cfg.input_dir = args.input_dir and Path(args.input_dir)  # type: ignore
    cfg.num_workers = args.num_workers  # type: ignore
    cfg.weight = args.weight  # type: ignore
    cfg.save_path = Path("models") / cfg.config_name  # type: ignore
    cfg.wandb = args.wandb  # type: ignore
    cfg.reproduce = args.reproduce  # type: ignore
    cfg.data_config = args.data_config  # type: ignore
    cfg.phase = "train"  # type: ignore


def configure_inference_args(cfg: Config, args: argparse.Namespace) -> None:
    cfg.inference.fold = args.fold  # type: ignore
    cfg.split.seed = args.seed  # type: ignore
    cfg.config_name = args.config  # type: ignore
    cfg.input_dir = args.input_dir and Path(args.input_dir)  # type: ignore
    cfg.input_path = args.input_path and Path(args.input_path)  # type: ignore
    cfg.num_workers = args.num_workers  # type: ignore
    cfg.weight = args.weight  # type: ignore
    if not cfg.weight:
        cfg.weight = cfg.config_name  # type: ignore
    cfg.inference.val_list_path = args.val_list_path and Path(args.val_list_path)  # type: ignore
    cfg.save_path = Path("models") / cfg.config_name  # type: ignore
    cfg.predict_dataset = args.predict_dataset  # type: ignore
    cfg.final = args.final  # type: ignore
    cfg.inference.num_tta = args.num_repetitions  # type: ignore
    cfg.reproduce = args.reproduce  # type: ignore
    cfg.out_path = args.out_path and Path(args.out_path)  # type: ignore
    cfg.data_config = None  # type: ignore
    cfg.phase = "inference"  # type: ignore


def configure_defaults(cfg: Config) -> None:
    if cfg.id_name is None:
        cfg.id_name = "utt_id"  # type: ignore


def configure_execution(cfg: Config) -> None:
    cfg.data_config = None  # type: ignore
    cfg.phase = "prediction"  # type: ignore
    cfg.print_config = False  # type: ignore
