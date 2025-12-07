from __future__ import annotations

from typing import TYPE_CHECKING

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
    import torch.nn as nn

    from utmosv2._settings._config import Config


class UTMOSv2Model(UTMOSv2ModelMixin):
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
        self._cfg_value = cfg

    @property
    def _cfg(self) -> Config:
        return self._cfg_value

    def eval(self) -> "nn.Module":
        return self._model.eval()

    def __call__(self, *args: Any, **kwargs: Any) -> "torch.Tensor":
        return self._model(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "_model":
            super().__setattr__(name, value)
        else:
            setattr(self._model, name, value)

    def __delattr__(self, name: str) -> None:
        delattr(self._model, name)

    def __repr__(self) -> str:
        return f"UTMOSv2Model({'('.join(self._model.__repr__().split('(')[1:])}"

    def __str__(self) -> str:
        return self.__repr__()

    def __dir__(self) -> list[str]:
        return super().__dir__() + self._model.__dir__()
