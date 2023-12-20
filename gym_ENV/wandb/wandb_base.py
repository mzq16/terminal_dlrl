from venv import logger
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from pathlib import Path
import wandb
import time
import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, sync_envs_normalization
from stable_baselines3.common.utils import safe_mean
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union
import cv2

class WandbCallback(BaseCallback):
    def __init__(self, cfg, vec_env, video_path, ckpt_dir, buffer_dir):
        super(WandbCallback, self).__init__(verbose=1)

        save_dir = Path.cwd()
        print(f"save_dir: {save_dir}")
        self._video_path = Path(video_path)
        self._video_path.mkdir(parents=True, exist_ok=True)
        self._ckpt_dir = Path(ckpt_dir)
        self._ckpt_dir.mkdir(parents=True, exist_ok=True)
        self._buffer_dir = Path(buffer_dir)
        self._buffer_dir.mkdir(parents=True, exist_ok=True)

        wandb.init(project=cfg['wb_project'], name=cfg['wb_name'], notes=cfg['wb_notes'], tags=cfg['wb_tags'])
        self.vec_env = vec_env
        
        # 一些eval参数
        self._eval_step = int(1e5)
        self._n_eval_episodes = 1
        self.warn = True
        self._render = False
        self._deterministic = True
        #self._buffer_step = int(1e4)
        self._save_step = int(1e3)
        self._save_buffer_step = int(5e3)

        self.episodes = 0

    def _init_callback(self):
        self.n_epoch = 0
        self._last_time_buffer = self.model.num_timesteps
        self._last_time_eval = self.model.num_timesteps
        self._last_time_save = self.model.num_timesteps
        self._last_time_save_buffer = self.model.num_timesteps
        wandb.log({
            "param/eval_step": self._eval_step,
            "param/n_eval_episodes": self._n_eval_episodes,
            "param/learning_rate": self.model.learning_rate,
            "param/n_envs": self.model.n_envs,
            "param/start_time": self.model.start_time,
            "param/start_numsteps": self.model._num_timesteps_at_start,
            "param/total_timesteps": self.model._total_timesteps,
            "param/stats_window_size": self.model._stats_window_size,
        })

    def _on_step(self) -> bool:
        # 被on_step调用，on_step在BaseCallback中，一般返回true，被rollout每一步调用
        
        infos = self.locals["infos"]
        info = infos[0]
        rewards = self.locals['rewards']
        dones = self.locals['dones']
        obs = self.locals['new_obs']
        
        if "reward_info" in info.keys() and info["reward_info"]["r_exc"] < -5:
            self._check_exc_situation()
        if dones[0] is True:
            if info["reward_info"]["r_exc"] > -5:
                print("checked done not exc")
            elif info["reward_info"]["r_arr"] < 10:
                print("checked done not arr")
            else:
                print(f"rewards:{rewards}, not right done situation")
            
        all_envs_dict = {"rollout/reward_avg_n_env": safe_mean(rewards)}
        # for i in range(self.model.n_envs):
        #    all_envs_dict[f'rollout/reward_{i}_env'] = rewards[i]
        #wandb.log(all_envs_dict, step=self.model.num_timesteps)
        for i in range(self.model.n_envs):
            all_envs_dict[f'rollout/reward_{i}_env'] = rewards[i]
            single_reward_info = infos[i].get('reward_info')
            all_envs_dict[f"rollout/reward_{i}_env/done"] = int(dones[i])
            for k, v in single_reward_info.items():
                all_envs_dict[f'rollout/reward_{i}_env/' + k] = v
            wandb.log(all_envs_dict, step=self.model.num_timesteps)

        assert self.model.ep_info_buffer is not None
        assert self.model.ep_success_buffer is not None
        if len(self.model.ep_info_buffer) > 0 and len(self.model.ep_info_buffer[0]) > 0:
            ep_reward_mean = safe_mean([ep_info["r"] for ep_info in self.model.ep_info_buffer])
            ep_length_mean = safe_mean([ep_info["l"] for ep_info in self.model.ep_info_buffer])
            wandb.log({
                "rollout/ep_reward_mean": ep_reward_mean,
                "rollout/ep_length_mean": ep_length_mean,
            }, step=self.model.num_timesteps)


        if self._eval_step > 0 and self.n_calls % self._eval_step == 0:
            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    ) from e

            # Reset success rate buffer
            self._is_success_buffer = []

            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.vec_env,
                n_eval_episodes=self._n_eval_episodes,
                render=self._render,
                deterministic=self._deterministic,
                return_episode_rewards=False,
                warn=self.warn,
                callback=self._log_success_callback,
            )
            wandb.log({
            "eval/ep_reward_mean": episode_rewards,
            "eval/ep_length_mean": episode_lengths,
        }, step=self.model.num_timesteps)
        
        return True

    def _on_training_start(self) -> None:
        # 父类有一个不是隐的，会调用这个，并且会加入local和global
        self.train_start_time = time.time()
        #print(f"training_local:{self.locals.keys()}")
        #print(f"training_globals:{self.globals.keys()}")
        # print('training start')
        
    def _on_rollout_start(self):
        # 同上，父类有一个非隐调用
        self.rollout_start_time = time.time()
        #print(f"rollout_local:{self.locals.keys()}")
        #print(f"rollout_globals:{self.globals.keys()}")
        #print('rollout start')

    def _on_training_end(self) -> None:
        # 这个地方是learn的最后
        # train time 其实是learn time，包括rollout的时间
        train_time = time.time() - self.train_start_time
        time_elapsed = time.time() - self.model.start_time

        # 最后的一些参数
        wandb.log({
            'time/fps': (self.model.num_timesteps-self.model._num_timesteps_at_start) / time_elapsed,
            'time/total_time_elapsed': time_elapsed,
            'time/total_learn_time': train_time,
        }, step=self.model.num_timesteps)

        # evaluate and save checkpoint
    
    def _on_rollout_end(self):
        rollout_time = time.time() - self.rollout_start_time
        logger_dict = {}
        logger_dict['rollout/time'] = rollout_time
        if "rollout/exploration_rate" in self.model.logger.name_to_value.keys():
            logger_dict["rollout/exploration_rate"] = self.model.logger.name_to_value["rollout/exploration_rate"]

        # 因为train 里面没有callback，所以将一些网络更新的东西放在这里
        # 这里表示rollout结束，train网络的开始，callback中的traing start不是train网络的开始，而是所有的开始
        if "train/loss" in self.model.logger.name_to_value.keys():
           logger_dict["train/loss"] = self.model.logger.name_to_value["train/loss"]
        if "train/current_q_values" in self.model.logger.name_to_value.keys():
            logger_dict["train/current_q_values"] = self.model.logger.name_to_value["train/current_q_values"]
        if "train/action_prob" in self.model.logger.name_to_value.keys():
            logger_dict["train/action_prob"] = self.model.logger.name_to_value["train/action_prob"]
        for k, v in self.model.logger.name_to_value.items():
            if k != "topo_pred" or k != "random":
                logger_dict[f"test_logger_dict/" + k] = v
        wandb.log(logger_dict, step=self.num_timesteps)

        # one episode
        if "time/episodes" in self.model.logger.name_to_value.keys():
            episodes = self.model.logger.name_to_value["time/episodes"]
            print(f"start eposiode {episodes}")
            if self.episodes != episodes:
                self.episodes = episodes
                episode_dict = {}
                for k, v in self.model.logger.name_to_value.items():
                    episode_dict[f"episodes/" + k] = v
                wandb.log(episode_dict, step=self.episodes)
        
    def _log_success_callback(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
        # copy了官方的Evalcallback
        """
        Callback passed to the  ``evaluate_policy`` function
        in order to log the success rate (when applicable),
        for instance when using HER.

        :param locals_:
        :param globals_:
        """
        info = locals_["info"]

        if locals_["done"]:
            maybe_is_success = info.get("is_success")
            if maybe_is_success is not None:
                self._is_success_buffer.append(maybe_is_success)

    def _check_exc_situation(self):
        infos = self.locals["infos"]
        info = infos[0]
        rewards = self.locals['rewards']
        dones = self.locals['dones']
        obs = self.locals['new_obs']
        if "random" in self.model.logger.name_to_value.keys():
                random_ = self.model.logger.name_to_value['random']
        else:
            random_ = None
        actions = self.locals['actions']
        cu_id, pr_id = info["ev_loc"]
        q_values = self.model.policy.q_net.q_values
        prob_values = self.model.policy.q_net.prob_values
        topo_mask = self.model.policy.q_net.topo_mask
        topo_pred = self.model.logger.name_to_value['topo_pred']
        evid_pred = self.model.logger.name_to_value['evid_pred']
        desid_pred = self.model.logger.name_to_value['desid_pred']
        print(f"q_value_net:{q_values}, prob_value_net:{prob_values}, topo_mask_net:{topo_mask}, \
                dones:{dones}, random_pred:{random_}, topo_pred:{topo_pred}, evid_pred:{evid_pred}, desid_pred:{desid_pred}")
        print(f"action:{actions}, curr_id:{cu_id}, prev_id:{pr_id}, \
                numstep:{self.num_timesteps}, obs_evid:{obs['ev_curr_id']}, obs_desid:{obs['des_id']}, obs_topomap:{obs['map_topo']}")
        arr_img = self.model.env.render()
        cv2.imwrite(f"img_{self.num_timesteps}.jpg", arr_img)


def init_callback_list(env, save_freq=100, save_path='./data/checkpoint_noimg/', save_replay_buffer=False, wandb_flag=True, verbose=2, cfg=None):
    '''
    verbose: 2就是什么信息都打印，0就都不打印，没1什么事，之后可以改
    '''
    callback_save = CheckpointCallback(save_freq=save_freq, save_path=save_path, verbose=verbose, save_replay_buffer=save_replay_buffer)
    if wandb_flag:
        if cfg is None:
            cfg = {
                'wb_project': "terminal_noimg_test",
                'wb_name': None,
                'wb_notes': None, 
                'wb_tags': None,
            }
        else:
            cfg = cfg
        callback_wandb = WandbCallback(cfg, env, "./data/vedeo", save_path, "./data/buffer")

    # 这个callback list 看了看源码应该是可以嵌套的，因为callback list也是继承了BaseCallback，
    # 嵌套就是[callback1, [callback2,callback0]]，
    # callback3 = [callback2,callback0]
    # 之后也是一一调用callback1._on_step，然后callback3._on_step-------> callback2._on_step, callback0._on_step
    # 没区别，不嵌套了，后面还有一个tqdm的callback，那个是把rollout和train放一块了，如果任务比较简单就和成一起吧
    # callback_list = CallbackList([callback_save, callback_wandb])
    if wandb_flag:
        callback_list = [callback_save, callback_wandb]
    else:
        callback_list = [callback_save]
    return callback_list