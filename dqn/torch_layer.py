from typing import Dict, List, Tuple, Type, Union

import gymnasium as gym
from gymnasium import spaces
import torch
from torch import nn
from stable_baselines3.common.preprocessing import get_flattened_obs_dim, is_image_space
from stable_baselines3.common.type_aliases import TensorDict
from stable_baselines3.common.utils import get_device
from stable_baselines3.dqn.policies import BaseFeaturesExtractor

def create_mlp(
    input_dim: int,
    output_dim: int,
    net_arch: List[int],
    activation_fn: Type[nn.Module] = nn.ReLU,
    with_bias: bool = True,
) -> List[nn.Module]:
    """
    Create a multi layer perceptron (MLP), which is
    a collection of fully-connected layers each followed by an activation function.

    :param input_dim: Dimension of the input vector
    :param output_dim:
    :param net_arch: Architecture of the neural net
        It represents the number of units per layer.
        The length of this list is the number of layers.
    :param activation_fn: The activation function
        to use after each layer.
    :param squash_output: useless
    :param with_bias: If set to False, the layers will not learn an additive bias
    :param map_info: whether contain the map topology infomation to mask the output
    :return:

    copy from sb3 and overide

    """

    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0], bias=with_bias), activation_fn()]
    else:
        modules = []
    
    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1], bias=with_bias))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim, bias=with_bias))
    
    return modules



# FeaturesExtractor
class myExtractor(BaseFeaturesExtractor):
    # copy and overide stable_baselines3.dqn.policies.CombinedExtractor
    def __init__(self, 
                 observation_space: spaces.Dict, 
                 features_dim: int = 1,
                 net_arch: list = [16, 16], 
                 activation_fn: Type[nn.Module] = nn.ReLU) -> None:
        super().__init__(observation_space, features_dim = features_dim)
        
        extractors: Dict[str, nn.Module] = {}

        total_concat_size = 0

        for key, subspace in observation_space.spaces.items():
            net_args = []
            if key != 'map_topo':
                if key != 'des_id':
                    net_args.append(nn.Flatten())
                    faltten_dim = get_flattened_obs_dim(subspace)
                    net_args.append(nn.Linear(faltten_dim, net_arch[0]))
                    for idx in range(len(net_arch) - 1):
                        net_args.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
                        net_args.append(activation_fn())
                    total_concat_size += net_arch[-1]
                else:
                    net_args.append(nn.Flatten())
                    faltten_dim = get_flattened_obs_dim(subspace)
                    net_args.append(nn.Linear(faltten_dim, 4))
                    net_args.append(activation_fn())
                    net_args.append(nn.Linear(4, 8))
                    net_args.append(activation_fn())
                    net_args.append(nn.Linear(8, 4))
                    net_args.append(activation_fn())
                    total_concat_size += 4
                extractors[key] = nn.Sequential(*net_args)

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations: TensorDict) -> torch.Tensor:
        encoded_tensor_list = []

        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        return torch.cat(encoded_tensor_list, dim=1)