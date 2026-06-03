"""Compatibility patches loaded automatically when this directory is on PYTHONPATH."""

import importlib.util
import sys


def _patch_guided_decoding_params(module):
    if hasattr(module, "GuidedDecodingParams"):
        return

    class GuidedDecodingParams:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    module.GuidedDecodingParams = GuidedDecodingParams


class _VllmSamplingParamsImportHook:
    _target = "vllm.sampling_params"

    def find_spec(self, fullname, path=None, target=None):
        if fullname != self._target:
            return None

        sys.meta_path = [hook for hook in sys.meta_path if hook is not self]
        spec = importlib.util.find_spec(fullname)
        if spec is None or spec.loader is None:
            return spec

        original_loader = spec.loader

        class _Loader:
            def create_module(self, spec):
                if hasattr(original_loader, "create_module"):
                    return original_loader.create_module(spec)
                return None

            def exec_module(self, module):
                original_loader.exec_module(module)
                _patch_guided_decoding_params(module)

        spec.loader = _Loader()
        return spec


already_loaded = sys.modules.get("vllm.sampling_params")
if already_loaded is not None:
    _patch_guided_decoding_params(already_loaded)
else:
    sys.meta_path.insert(0, _VllmSamplingParamsImportHook())

