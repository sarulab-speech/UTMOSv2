from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import TypeAlias

# NOTE: Python 3.12 introduces the type statement, so once Python 3.11 is dropped,
# it should be updated to use that instead.
Config: TypeAlias = SimpleNamespace | ModuleType


def configure_args(cfg: Config, args):
    cfg.fold = args.fold
    cfg.split.seed = args.seed
    cfg.config_name = args.config
    cfg.input_dir = args.input_dir and Path(args.input_dir)
    cfg.num_workers = args.num_workers
    cfg.weight = args.weight
    cfg.save_path = Path("models") / cfg.config_name
    cfg.wandb = args.wandb
    cfg.reproduce = args.reproduce
    cfg.data_config = args.data_config
    cfg.phase = "train"


def configure_inference_args(cfg: Config, args):
    cfg.inference.fold = args.fold
    cfg.split.seed = args.seed
    cfg.config_name = args.config
    cfg.input_dir = args.input_dir and Path(args.input_dir)
    cfg.input_path = args.input_path and Path(args.input_path)
    cfg.num_workers = args.num_workers
    cfg.weight = args.weight
    if not cfg.weight:
        cfg.weight = cfg.config_name
    cfg.inference.val_list_path = args.val_list_path and Path(args.val_list_path)
    cfg.save_path = Path("models") / cfg.config_name
    cfg.predict_dataset = args.predict_dataset
    cfg.final = args.final
    cfg.inference.num_tta = args.num_repetitions
    cfg.reproduce = args.reproduce
    cfg.out_path = args.out_path and Path(args.out_path)
    cfg.data_config = None
    cfg.phase = "inference"


def configure_defaults(cfg: Config):
    if cfg.id_name is None:
        cfg.id_name = "utt_id"


def configure_execution(cfg: Config):
    cfg.data_config = None
    cfg.phase = "prediction"
    cfg.print_config = False
