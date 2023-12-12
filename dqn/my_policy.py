from typing import Any, Dict, List, Optional, Type
from gymnasium import spaces
import stable_baselines3 as sb3
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.dqn.policies import DQNPolicy, QNetwork
from stable_baselines3.dqn.dqn import DQN
from gymnasium import spaces
import torch as th
from torch import nn
import torch
from .torch_layer import myExtractor
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
import copy
import numpy as np

        
class myQNetwork(QNetwork):
    

    def __init__(self, 
                 observation_space: spaces.Space, 
                 action_space: spaces.Discrete, 
                 features_extractor: BaseFeaturesExtractor, 
                 features_dim: int, 
                 net_arch: Optional[List[int]] = None, 
                 activation_fn: Type[nn.Module] = nn.ReLU, 
                 normalize_images: bool = True) -> None:
        super().__init__(observation_space, action_space, features_extractor, features_dim, net_arch, activation_fn, normalize_images)
        # override init to add softmax layer
        self.softmax = nn.Softmax(dim=1)    # (batch_size, output)
        self._prob_values = None
        self._q_values = None
        self.topo_bias = None
        
    def forward(self, obs: PyTorchObs) -> th.Tensor:
        """
        Predict the q-values.

        :param obs: Observation
        :return: The estimated Q-Value for each action.

        override，其实这个函数不override也是可以的，因为override 的部分移动到下面了
        """
        q_values = self.q_net(self.extract_features(obs, self.features_extractor))
        return q_values
    
    def _predict(self, observation: PyTorchObs, deterministic: bool = True, bios_threshold: float = 10000.0) -> th.Tensor:
        # override
        q_values = self(observation).detach()
        self._q_values = q_values.clone().detach()
        # topo mask
        assert "map_topo" in observation.keys(), "no map topo info"
        topo_bias = observation['map_topo']
        self.topo_bias = topo_bias.clone().detach()
        q_values_masked = q_values - topo_bias * bios_threshold
        prob_values = self.softmax(q_values_masked)

        # Greedy action
        action = prob_values.argmax(dim=1).reshape(-1)
        self._prob_values = prob_values.clone().detach()
        return action
    
    @property
    def prob_values(self) -> np.ndarray:
        return self._prob_values.detach().cpu().numpy() if self._prob_values is not None else None
    
    @property
    def q_values(self) -> np.ndarray:
        return self._q_values.detach().cpu().numpy() if self._q_values is not None else None
    
    @ property
    def topo_mask(self):
        return self.topo_mask.detach().cpu().numpy() if self.topo_bias is not None else None
    
class myPolicy(DQNPolicy):
    def __init__(self, observation_space: spaces.Space, 
                 action_space: spaces.Discrete, 
                 lr_schedule: Schedule, 
                 net_arch: Optional[List[int]] = None, 
                 activation_fn: Type[nn.Module] = nn.ReLU, 
                 features_extractor_class: Type[BaseFeaturesExtractor] = myExtractor, 
                 features_extractor_kwargs: Optional[Dict[str, Any]] = None, 
                 normalize_images: bool = True, 
                 optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam, 
                 optimizer_kwargs: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(observation_space, action_space, lr_schedule, net_arch, activation_fn, 
                         features_extractor_class, features_extractor_kwargs, normalize_images, 
                         optimizer_class, optimizer_kwargs)
        self.net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "net_arch": self.net_arch,
            "activation_fn": self.activation_fn,
        }
        
    def make_q_net(self) -> myQNetwork:
        # Make sure we always have separate networks for features extractors etc
        # override
        net_args = self._update_features_extractor(self.net_args, features_extractor=self.features_extractor)
        return myQNetwork(**net_args).to(self.device)