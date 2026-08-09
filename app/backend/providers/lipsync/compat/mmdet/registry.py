"""MMEngine model registry for the local CSPNeXt implementation."""

from mmengine.registry import MODELS as MMENGINE_MODELS
from mmengine.registry import Registry

MODELS = Registry("model", parent=MMENGINE_MODELS, locations=["mmdet.models"])

__all__ = ["MODELS"]
