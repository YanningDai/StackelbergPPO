from typing import List, Union, Dict, Optional
import torch
import torch.nn as nn
from design_opt.models.bodygen_policy import BodyGenPolicy
from design_opt.models.bodygen_critic import BodyGenValue
import os
import time
import tempfile
from contextlib import contextmanager

class ParameterManager:

    def __init__(self, cfg):
        self.cfg = cfg
        self._init_part_mappings() 
        
    def _init_part_mappings(self):
        self.norms = {
            'skel_trans': ['skel_norm'],
            'attr_trans': ['attr_norm'],
            'execution': ['control_norm']
        }
        self.base_mapping = {
            'skel_trans': ['skel_transformer', 'skel_mlp'],
            'attr_trans': ['attr_transformer', 'attr_mlp'],
            'execution': ['control_transformer', 'control_mlp']
        }
        
        self.policy_specific = {
            'skel_trans': ['skel_action_logits'],
            'attr_trans': ['attr_action_mean', 'attr_action_log_std'],
            'execution': ['control_action_mean', 'control_action_log_std']
        }
        
        self.critic_specific = {
            'skel_trans': ['skel_value_head'],
            'attr_trans': ['attr_value_head'],
            'execution': ['control_value_head']
        }
        
        self.policy_mapping = {k: self.norms[k] + self.base_mapping[k] + self.policy_specific[k] for k in self.base_mapping}
        self.critic_mapping = {k: self.norms[k] + self.base_mapping[k] + self.critic_specific[k] for k in self.base_mapping}

    def _get_mapping(self, model: Union[BodyGenPolicy, BodyGenValue]) -> Dict:
        if isinstance(model, BodyGenPolicy):
            return self.policy_mapping
        elif isinstance(model, BodyGenValue):
            return self.critic_mapping
        else:
            raise TypeError(f"Unsupported model type: {type(model)}")
    
    def _validate_part_names(self, model: Union[BodyGenPolicy, BodyGenValue], part_names: List[str]) -> None:
        if not isinstance(part_names, list):
            raise TypeError(f"part_names must be list, got {type(part_names)}")
        
        valid_parts = set(self._get_mapping(model).keys())
        for part in part_names:
            if part not in valid_parts:
                raise ValueError(f"Invalid part name: {part}. Valid options: {list(valid_parts)}")
            
    def get_parameter_names(self, model: Union[BodyGenPolicy, BodyGenValue], part_names: List[str]) -> List[str]:
        self._validate_part_names(model, part_names)
        current_mapping = self._get_mapping(model)
        param_names = []
        
        for part_name in part_names:
            for attr in current_mapping[part_name]:
                module = getattr(model, attr, None)
                if module is None:
                    continue
                
                if hasattr(module, 'named_parameters'):
                    for name, _ in module.named_parameters(prefix=attr):
                        param_names.append(name)
                else:
                    param_names.append(attr)
        
        return param_names

    
    def selective_backward(self, loss: torch.Tensor, model: Union[BodyGenPolicy, BodyGenValue], 
                          update_list: Optional[List[str]] = None) -> None:
        if update_list is None: 
            loss.backward()
            return
            
        update_params = self.get_parameter_names(model, update_list)
        
        loss.backward()
        
        for name, param in model.named_parameters():
            if name not in update_params and param.grad is not None:
                param.grad = None
        
    @contextmanager
    def norm_mode_manager(self, policy_net, value_net, update_list=None):
        if update_list is not None:
            for net in [policy_net, value_net]:
                for norm_names in self.norms.values():
                    for norm_name in norm_names:
                        norm_layer = getattr(net, norm_name, None)
                        if norm_layer is not None: 
                            norm_layer.eval()
                
                for module in update_list:
                    for norm_name in self.norms.get(module, []):
                        norm_layer = getattr(net, norm_name, None)
                        if norm_layer is not None:
                            norm_layer.train()
        
        yield
        
        for net in [policy_net, value_net]:
            for norm_names in self.norms.values():
                for norm_name in norm_names:
                    norm_layer = getattr(net, norm_name, None)
                    if norm_layer is not None: 
                        norm_layer.train()
    
    def get_param_tensors(self, model, part_names):
        names = set(self.get_parameter_names(model, part_names))
        return [p for n, p in model.named_parameters() if n in names]
