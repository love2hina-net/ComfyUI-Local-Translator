import os
from typing import (
    Any,
    Union,
)

import torch
from transformers import (
    AutoModelForCausalLM,
    LlamaForCausalLM,
)

from comfy import model_management

from ..common import getLogger

logger = getLogger()

class ProxyForLM:
    _MODEL: LlamaForCausalLM
    _SIZE: int
    _LOAD_DEVICE: torch.device
    _OFFLOAD_DEVICE: torch.device

    def __init__(self, model: LlamaForCausalLM):
        self._MODEL = model
        self._SIZE = model_management.module_size(model)
        self._LOAD_DEVICE = model.device
        self._OFFLOAD_DEVICE = model_management.text_encoder_offload_device()
        logger.debug(f"load device: {self._LOAD_DEVICE}, offload device: {self._OFFLOAD_DEVICE}, dtype: {self.model_dtype()}")

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike[str]],
        *model_args,
        **kwargs,
    ) -> 'ProxyForLM':
        load_device = model_management.text_encoder_device()
        model: LlamaForCausalLM = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            **kwargs,
        )
        model.to(device=load_device)
        return ProxyForLM(model)

    def is_clone(self, other: Any) -> bool:
        return (hasattr(other, 'model') and self.model is other.model)

    @property
    def parent(self) -> None:
        return None
    
    @property
    def model(self) -> LlamaForCausalLM:
        return self._MODEL
    
    def model_dtype(self):
        return self.model.dtype

    @property
    def load_device(self):
        return self._LOAD_DEVICE
    
    @property
    def offload_device(self):
        return self._OFFLOAD_DEVICE
    
    def current_loaded_device(self):
        return self.model.device

    def model_size(self):
        return self._SIZE

    def loaded_size(self):
        if (self.model.device.type == self._LOAD_DEVICE.type) and (self.model.device.index == self._LOAD_DEVICE.index):
            return self._SIZE
        else:
            return 0

    def model_patches_to(self, device):
        logger.debug(f"model_patches_to: {device}")

        if (isinstance(device, torch.device)):
            self.model.to(device=device)
        else:
            logger.debug(f"Ignore cast to: {device}")

    def model_patches_models(self):
        return []

    def partially_load(self, device_to, extra_memory=0, force_patch_weights=False):
        logger.debug(f"partially_load: {device_to}")
        self.model.to(device=device_to)
        return self.loaded_size()
    
    def partially_unload(self, device_to, memory_to_free=0):
        logger.debug(f"partially_unload: {device_to}")
        self.model.to(device=device_to)
        return self.loaded_size()
    
    def detach(self, unpatch_all=True):
        self.model_patches_to(self._OFFLOAD_DEVICE)
