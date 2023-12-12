import stable_baselines3 as sb3
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3.dqn.policies import DQNPolicy
from stable_baselines3.dqn.dqn import DQN
import warnings
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type, TypeVar, Union
import torch as th
from torch.nn import functional as F
import numpy as np


# 其实learn函数最好也重写一下，有一些参数get不到。（offpolicy）
class my_dqn(DQN):
    def __init__(self, 
                 policy: Union[str, Type[DQNPolicy]], 
                 env: Union[GymEnv, str], 
                 learning_rate: Union[float, Schedule] = 1e-4, 
                 buffer_size: int = 1000000, 
                 learning_starts: int = 50000, 
                 batch_size: int = 32, 
                 tau: float = 1, 
                 gamma: float = 0.99, 
                 train_freq: Union[int, Tuple[int, str]] = 4, 
                 gradient_steps: int = 1, 
                 replay_buffer_class: Optional[Type[ReplayBuffer]] = None, 
                 replay_buffer_kwargs: Optional[Dict[str, Any]] = None, 
                 optimize_memory_usage: bool = False, 
                 target_update_interval: int = 10000, 
                 exploration_fraction: float = 0.1, 
                 exploration_initial_eps: float = 1, 
                 exploration_final_eps: float = 0.05, 
                 max_grad_norm: float = 10, 
                 stats_window_size: int = 100, 
                 tensorboard_log: Optional[str] = None, 
                 policy_kwargs: Optional[Dict[str, Any]] = None, 
                 verbose: int = 0, 
                 seed: Optional[int] = None, 
                 device: Union[th.device, str] = "auto", 
                 _init_setup_model: bool = True) -> None:
        super().__init__(policy, env, learning_rate, buffer_size, learning_starts, batch_size, tau, gamma, 
                         train_freq, gradient_steps, replay_buffer_class, replay_buffer_kwargs, optimize_memory_usage, 
                         target_update_interval, exploration_fraction, exploration_initial_eps, exploration_final_eps,
                         max_grad_norm, stats_window_size, tensorboard_log, policy_kwargs, verbose, seed, device, _init_setup_model)
        
    def predict(self, 
                observation: Union[np.ndarray, Dict[str, np.ndarray]], 
                state: Optional[Tuple[np.ndarray, ...]] = None, 
                episode_start: Optional[np.ndarray] = None, 
                deterministic: bool = False
                ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        # overide the dqn and base_class predict function to include our epsilon-greedy exploration with topo mask
        
        assert "map_topo" in observation.keys()       
        
        if not deterministic and np.random.rand() < self.exploration_rate:
            if self.policy.is_vectorized_observation(observation):
                if isinstance(observation, dict):
                    n_batch = observation[next(iter(observation.keys()))].shape[0]
                else:
                    n_batch = observation.shape[0]
                a_list = []
                for i in range(n_batch):
                    topo_info = observation["map_topo"][i]
                    indice = np.where(topo_info < 0.5)[0]
                    a_list.append(np.random.choice(indice))
                action = np.array(a_list)
        else:
            action, state = self.policy.predict(observation, state, episode_start, deterministic)
        return action, state
        
    def _on_step(self) -> None:
        super()._on_step()
        # batch size,其实不是batchsize了，而是number env
        action_probs = self.policy.q_net.prob_values
        q_values = self.policy.q_net.q_values
        if action_probs is not None and q_values is not None:
            n_env = action_probs.shape[0]
            action_entropy = 0
            q_value_avg = 0
            for i in range(n_env):
                action_entropy += self.cal_entropy(action_probs[i])
                q_value_avg += max(q_values[i])
                action_entropy = action_entropy / n_env
                q_value_avg = q_value_avg / n_env
                self.logger.record(f"train/action_prob_{i}", action_entropy)
                self.logger.record(f"train/q_value_qnet_{i}", q_value_avg)

    def cal_entropy(self, action_prob: np.ndarray):
        action_prob_revise = action_prob[action_prob != 0]
        return -np.sum(action_prob_revise * np.log2(action_prob_revise))

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]

            with th.no_grad():
                # Compute the next Q-values using the target network
                next_q_values = self.q_net_target(replay_data.next_observations)
                # Follow greedy policy: use the one with the highest value
                next_q_values, _ = next_q_values.max(dim=1)
                # Avoid potential broadcast issue
                next_q_values = next_q_values.reshape(-1, 1)
                # 1-step TD target
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values estimates
            current_q_values = self.q_net(replay_data.observations)

            # Retrieve the q-values for the actions from the replay buffer
            current_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions.long())

            # Compute Huber loss (less sensitive to outliers)
            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())

            # Optimize the policy
            self.policy.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()
            

        # Increase update counter
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))
        self.logger.record("train/current_q_values", np.mean(current_q_values.cpu().detach().numpy()))