import torch
import torch.nn as nn
from typing import cast

from utmosv2._settings._config import Config
from utmosv2.dataset._utils import get_dataset_num
from utmosv2.model import MultiSpecExtModel, MultiSpecModelV2, SSLExtModel


class SSLMultiSpecExtModelV1(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.ssl = SSLExtModel(cfg)
        self.spec_long = MultiSpecModelV2(cfg)
        self.ssl.load_state_dict(
            torch.load(
                f"outputs/{cfg.model.ssl_spec.ssl_weight}/fold{cfg.now_fold}_s{cfg.split.seed}_best_model.pth"
            )
        )
        self.spec_long.load_state_dict(
            torch.load(
                f"outputs/{cfg.model.ssl_spec.spec_weight}/fold{cfg.now_fold}_s{cfg.split.seed}_best_model.pth"
            )
        )
        if cfg.model.ssl_spec.freeze:
            for param in self.ssl.parameters():
                param.requires_grad = False
            for param in self.spec_long.parameters():
                param.requires_grad = False
        self.ssl.fc = nn.Identity()
        self.spec_long.fc = nn.Identity()

        self.num_dataset = get_dataset_num(cfg)

        self.fc = nn.Linear(
            cast(int, self.ssl.fc.in_features)
            + cast(int, self.spec_long.fc.in_features)
            + self.num_dataset,
            cfg.model.ssl_spec.num_classes,
        )

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor, d: torch.Tensor
    ) -> torch.Tensor:
        x1 = self.ssl(x1, torch.zeros(x1.shape[0], self.num_dataset).to(x1.device))
        x2 = self.spec_long(x2)
        x = torch.cat([x1, x2, d], dim=1)
        x = self.fc(x)
        return x


class SSLMultiSpecExtModelV2(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.ssl = SSLExtModel(cfg)
        self.spec_long = MultiSpecExtModel(cfg)
        if cfg.model.ssl_spec.ssl_weight is not None and cfg.phase == "train":
            self.ssl.load_state_dict(
                torch.load(
                    f"outputs/{cfg.model.ssl_spec.ssl_weight}/fold{cfg.now_fold}_s{cfg.split.seed}_best_model.pth"
                )
            )
        if cfg.model.ssl_spec.spec_weight is not None and cfg.phase == "train":
            self.spec_long.load_state_dict(
                torch.load(
                    f"outputs/{cfg.model.ssl_spec.spec_weight}/fold{cfg.now_fold}_s{cfg.split.seed}_best_model.pth"
                )
            )
        if cfg.model.ssl_spec.freeze:
            for param in self.ssl.parameters():
                param.requires_grad = False
            for param in self.spec_long.parameters():
                param.requires_grad = False
        self.ssl.fc = nn.Identity()
        self.spec_long.fc = nn.Identity()

        self.num_dataset = get_dataset_num(cfg)

        self.fc = nn.Linear(
            cast(int, self.ssl.fc.in_features)
            + cast(int, self.spec_long.fc.in_features)
            + self.num_dataset,
            cfg.model.ssl_spec.num_classes,
        )

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor, d: torch.Tensor
    ) -> torch.Tensor:
        x1 = self.ssl(x1, torch.zeros(x1.shape[0], self.num_dataset).to(x1.device))
        x2 = self.spec_long(
            x2, torch.zeros(x1.shape[0], self.num_dataset).to(x1.device)
        )
        x = torch.cat([x1, x2, d], dim=1)
        x = self.fc(x)
        return x
