from types import ModuleType, SimpleNamespace

from utmosv2._core.model._common import UTMOSv2ModelMixin
from utmosv2.model import (
    MultiSpecExtModel,
    MultiSpecModelV2,
    SSLExtModel,
    SSLMultiSpecExtModelV1,
    SSLMultiSpecExtModelV2,
)


class UTMOSv2Model(UTMOSv2ModelMixin):
    def __init__(self, cfg: SimpleNamespace | ModuleType):
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
    def _cfg(self):
        return self._cfg_value

    def eval(self):
        return self._model.eval()

    def __call__(self, *args, **kwargs):
        return self._model(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._model, name)

    def __setattr__(self, name, value):
        if name == "_model":
            super().__setattr__(name, value)
        else:
            setattr(self._model, name, value)

    def __delattr__(self, name):
        delattr(self._model, name)

    def __repr__(self):
        return f"UTMOSv2Model({'('.join(self._model.__repr__().split('(')[1:])}"

    def __str__(self):
        return self.__repr__()

    def __dir__(self):
        return super().__dir__() + dir(self._model)
