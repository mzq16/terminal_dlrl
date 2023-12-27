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
                 cnn_output_dim: int = 256, 
                 normalized_image: bool = False,
                 activation_fn: Type[nn.Module] = nn.ReLU) -> None:
        super().__init__(observation_space, features_dim = features_dim)
        
        extractors: Dict[str, nn.Module] = {}

        total_concat_size = 0

        for key, subspace in observation_space.spaces.items():
            net_args = []
            if is_image_space(subspace, normalized_image=normalized_image):
                    extractors[key] = NatureCNN(subspace, features_dim=cnn_output_dim, normalized_image=normalized_image)
                    total_concat_size += cnn_output_dim
            elif key != 'map_topo':
                if key == 'des_id':
                    net_args.append(nn.Flatten())
                    faltten_dim = get_flattened_obs_dim(subspace)
                    net_args.append(nn.Linear(faltten_dim, 4))
                    net_args.append(activation_fn())
                    net_args.append(nn.Linear(4, 8))
                    net_args.append(activation_fn())
                    net_args.append(nn.Linear(8, 4))
                    net_args.append(activation_fn())
                    total_concat_size += 4
                else:
                    net_args.append(nn.Flatten())
                    faltten_dim = get_flattened_obs_dim(subspace)
                    net_args.append(nn.Linear(faltten_dim, net_arch[0]))
                    for idx in range(len(net_arch) - 1):
                        net_args.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
                        net_args.append(activation_fn())
                    total_concat_size += net_arch[-1]
                extractors[key] = nn.Sequential(*net_args)

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations: TensorDict) -> torch.Tensor:
        encoded_tensor_list = []

        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        return torch.cat(encoded_tensor_list, dim=1)
    
class NatureCNN(BaseFeaturesExtractor):
    """
    copy from sb3

    :param observation_space:
    :param features_dim: Number of features extracted.
        This corresponds to the number of unit for the last layer.
    :param normalized_image: Whether to assume that the image is already normalized
        or not (this disables dtype and bounds checks): when True, it only checks that
        the space is a Box and has 3 dimensions.
        Otherwise, it checks that it has expected dtype (uint8) and bounds (values in [0, 255]).
    """

    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 512,
        normalized_image: bool = False,
    ) -> None:
        assert isinstance(observation_space, spaces.Box), (
            "NatureCNN must be used with a gym.spaces.Box ",
            f"observation space, not {observation_space}",
        )
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        assert is_image_space(observation_space, check_channels=False, normalized_image=normalized_image), (
            "You should use NatureCNN "
        )
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 8, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            #nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(8, 16, kernel_size=5, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, 1024), 
            nn.ReLU(),
            nn.Linear(1024, 512), 
            nn.ReLU(),
            nn.Linear(512, features_dim), 
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))