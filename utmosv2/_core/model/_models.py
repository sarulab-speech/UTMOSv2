from __future__ import annotations

from typing import TYPE_CHECKING

import torch.nn as nn

from utmosv2._core.model._common import UTMOSv2ModelMixin
from utmosv2.model import (
    MultiSpecExtModel,
    MultiSpecModelV2,
    SSLExtModel,
    SSLMultiSpecExtModelV1,
    SSLMultiSpecExtModelV2,
)

if TYPE_CHECKING:
    from typing import Any

    import torch

    from utmosv2._settings._config import Config


class UTMOSv2Model(nn.Module, UTMOSv2ModelMixin):
    """
    UTMOSv2Model class that wraps different models specified by the configuration.
    This class allows for flexible model selection and provides a unified interface for evaluation, calling, and prediction.
    """

    def __init__(self, cfg: Config) -> None:
        """
        Initialize the UTMOSv2Model with a specified configuration.

        Args:
            cfg (SimpleNamespace | ModuleType): Configuration object that contains the model configuration.

        Raises:
            ValueError: If the model name specified in the configuration is not recognized.
        """
        super().__init__()
        models = {
            "multi_spec_ext": MultiSpecExtModel,
            "multi_specv2": MultiSpecModelV2,
            "sslext": SSLExtModel,
            "ssl_multispec_ext": SSLMultiSpecExtModelV1,
            "ssl_multispec_ext_v2": SSLMultiSpecExtModelV2,
        }
        if cfg.model.name not in models:
            raise ValueError(f"Unknown model name: {cfg.model.name}")
        self._model = models[cfg.model.name](cfg)
        self._cfg = cfg

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self._model(*args, **kwargs)

    def state_dict(self, *args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:  # type: ignore
        return self._model.state_dict(*args, **kwargs)

    def load_state_dict(  # type: ignore
        self,
        state_dict: dict[str, torch.Tensor],
        **kwargs: Any,
    ) -> Any:
        return self._model.load_state_dict(state_dict, **kwargs)

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)  # type: ignore[attr-defined]
        except AttributeError:
            return getattr(self._model, name)

    def __repr__(self) -> str:
        return f"UTMOSv2Model({'('.join(self._model.__repr__().split('(')[1:])}"

    def __str__(self) -> str:
        return self.__repr__()

    def __dir__(self) -> list[str]:
        return sorted(set(super().__dir__() + dir(self._model)))
