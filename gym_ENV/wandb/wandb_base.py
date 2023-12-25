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
import copy
from collections import defaultdict

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
        # self.vec_env = vec_env
        eval_env = copy.deepcopy(vec_env)
        if not isinstance(eval_env, VecEnv):
            eval_env = DummyVecEnv([lambda: eval_env])
        self.eval_env = eval_env
        # 一些eval参数
        self._eval_step = int(1e4)
        self._n_eval_episodes = 2
        self.warn = True
        self._render = False
        self._deterministic = True
        self.episodes = 0
        self.all_envs_dict = defaultdict(list)

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
        # 此外由于on step都上传wandb，数据有一些太多了，导致wandb打开特别慢
        # 所以这里就统计，用list存下来，上传放到rollout end
        infos = self.locals["infos"]
        info = infos[0]
        rewards = self.locals['rewards']
        dones = self.locals['dones']
        obs = self.locals['new_obs']
        
        # 异常点检测，None检测
        if info["reward_info"]["r_exc"] < -2:
            self._check_exc_situation()
        if dones[0] is True:
            if info["reward_info"]["r_exc"] > -2:
                print("checked done not exc")
            elif info["reward_info"]["r_arr"] < 3:
                print("checked done not arr")
            else:
                print(f"rewards:{rewards}, not right done situation")
        
        # 统计上传wandb的
        # self.all_envs_dict["rollout/reward_avg_n_env"].append(safe_mean(rewards))     #只有一个环境
        action_probs = self.model.policy.q_net.prob_values
        q_values = self.model.policy.q_net.q_values
        for i in range(self.model.n_envs):
            if i > 2:
                break
            self.all_envs_dict[f'rollout/reward_{i}_env'].append(rewards[i])
            single_reward_info = infos[i].get('reward_info')
            self.all_envs_dict[f"rollout/reward_{i}_env/done"].append(int(dones[i])) 
            for k, v in single_reward_info.items():
                self.all_envs_dict[f'rollout/reward_{i}_env/' + k].append(v) 
            if action_probs is not None:
                self.all_envs_dict[f"train/action_prob_{i}"].append(self.cal_entropy(action_probs[i]))
            if q_values is not None:
                self.all_envs_dict[f"train/q_value_qnet_{i}"].append(max(q_values[i]))
            # wandb.log(all_envs_dict, step=self.model.num_timesteps)
                
        return True

    def _on_training_start(self) -> None:
        # 父类有一个不是隐的，会调用这个，并且会加入local和global
        self.train_start_time = time.time()
        #print(f"training_local:{self.locals.keys()}")
        #print(f"training_globals:{self.globals.keys()}")
        # print('training start')
        
    def _on_rollout_start(self):
        # 同上，父类有一个非隐调用
        # train 没有end，所以把这个evaluate放这里，放step那里，每次rollout都得调用
        # 破案了，evaluate之后就会出现None的情况，大概率是没有reset
        self.rollout_start_time = time.time()

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
        # 交给evaluate callback了
    
    def _on_rollout_end(self):
        rollout_time = time.time() - self.rollout_start_time
        if self.model.num_timesteps % self._eval_step == 0:
            self._evaluate()
        logger_dict = {}
        logger_dict['rollout/time'] = rollout_time
        if "rollout/exploration_rate" in self.model.logger.name_to_value.keys():
            logger_dict["rollout/exploration_rate"] = self.model.logger.name_to_value["rollout/exploration_rate"]

        # get rollout param history
        for k, v in self.all_envs_dict.items():
            if k.split('/')[0] != 'debug': # 除debug的以外，上传
                logger_dict[k] = safe_mean(v)
        
        # get ep log
        assert self.model.ep_info_buffer is not None
        assert self.model.ep_success_buffer is not None
        if len(self.model.ep_info_buffer) > 0 and len(self.model.ep_info_buffer[0]) > 0:
            ep_reward_mean = safe_mean([ep_info["r"] for ep_info in self.model.ep_info_buffer])
            ep_length_mean = safe_mean([ep_info["l"] for ep_info in self.model.ep_info_buffer])
            logger_dict["rollout/ep_reward_mean"] = ep_reward_mean
            logger_dict["rollout/ep_length_mean"] = ep_length_mean

        # 因为train 里面没有callback，所以将一些网络更新的东西放在这里
        # 这里表示rollout结束，train网络的开始，callback中的traing start不是train网络的开始，而是所有的开始
        # 这下面的代码if什么的可以用dict中的get
        
        logger_dict["train/loss"] = self.model.logger.name_to_value.get("train/loss")
        logger_dict["train/current_q_values"] = self.model.logger.name_to_value.get("train/current_q_values")
        wandb.log(logger_dict, step=self.num_timesteps)

        # one episode
        # 至今好像还没有episode的数据
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
        cu_id, pr_id = info["ev_loc"]
        rewards = self.locals['rewards']
        dones = self.locals['dones']
        new_obs = self.locals['new_obs']
        actions = self.locals['actions']
        obs = self.model._last_obs      # new obs就是新的episode了，这个是旧的倒数第二帧
        history_id = obs.get("history_id")
        q_values = self.model.policy.q_net.q_values
        prob_values = self.model.policy.q_net.prob_values
        topo_mask = self.model.policy.q_net.topo_mask
        random_ = self.model.logger.name_to_value.get("debug/random")
        topo_pred = self.model.logger.name_to_value.get('debug/topo_pred')
        history_pred = self.model.logger.name_to_value.get("debug/history_pred")    # 原本有的，后来变成history了
        desid_pred = self.model.logger.name_to_value.get('debug/desid_pred')
        print(f"policy_value: q_value_net:{q_values}, prob_value_net:{prob_values}, topo_mask_net:{topo_mask}")
        print(f"logger: topo_pred:{topo_pred}, history_pred:{history_pred}, desid_pred:{desid_pred}, random_pred:{random_}")
        print(f"locals: action:{actions}, dones:{dones}, reward:{rewards}, curr_id:{cu_id}, prev_id:{pr_id}, \
              numstep:{self.num_timesteps}, obs_desid:{obs['des_id']}, obs_topomap:{obs['map_topo']}")
        print(f"history_id:{history_id}")
        arr_img = self.model.env.render()
        cv2.imwrite(f"img_{self.num_timesteps}_new.jpg", arr_img)

    def _evaluate(self):
        # n_calls其实在trian那边，_n_updates的更新跟着gradient step走，其实可以放在rollout那边，start或end都行，因为roll的时候网络这边不变
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
                self.eval_env,
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

    def cal_entropy(self, action_prob: np.ndarray):
        action_prob_revise = action_prob[action_prob != 0]
        return -np.sum(action_prob_revise * np.log2(action_prob_revise))

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
        callback_wandb = WandbCallback(cfg, env, "./data/video", save_path, "./data/buffer")

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