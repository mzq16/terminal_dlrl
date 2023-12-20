
import copy
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv
import torch
from tqdm import tqdm
from gym_ENV.wandb.wandb_base import init_callback_list
import cv2
from dqn.my_dqn import my_dqn
from dqn.my_policy import myPolicy
from stable_baselines3.common.logger import configure
import os
from datetime import datetime
import numpy as np

class myrobot(object):
    def __init__(self, base_path = './data/checkpoint_noimg/', 
                 buffer_step = None, ckpt_step = None, policy_kwargs = {},
                 wandb_kwargs = None):
        self.base_path = base_path
        if not os.path.exists(base_path):
            os.mkdir(base_path)
        self.policy_kwargs = policy_kwargs
        self.wandb_kwargs = wandb_kwargs
        buffer_path, buffer_timestep = self.get_buffer_name(buffer_step)
        if buffer_path is None:
            print("no buffer to load")
        else:
            self.latest_buffer_path = os.path.join(base_path, buffer_path)
            print(f"load {buffer_timestep} buffer")
        ckpt_path, ckpt_timestep = self.get_ckpt_name(ckpt_step)
        if ckpt_path is None:
            print("no ckpt to load, start new training")
        else:
            self.latest_ckpt_path = os.path.join(base_path, ckpt_path)
            print(f"load {ckpt_timestep} ckpt")
        self.env = None
        self.callback = None
        self._last_obs = None

    def set_env(self):
        env = gym.make('Terminal_Env-v0', num_vehicle = 10, map_size = [1200, 700], 
               render_mode = 'rgb_array', seed = 24, text_width = 400,
               model_arg = {'input_dim': 2, 'hidden_dim': 64, 'output_dim': 2, 'num_layers': 2,},
            )
        self.env = env
        return env
    
    def set_model(self, env, wandb_flag = True):
        tmp_path = "./tmp/sb3_log/"
        # set up logger
        new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
        
        callback_list = init_callback_list(env = env, save_path=self.base_path, save_freq = 5e3, save_replay_buffer=True, 
                                           verbose=0, wandb_flag=wandb_flag, cfg=self.wandb_kwargs)
        
        model = my_dqn(myPolicy, env, verbose=0, buffer_size=100000, learning_starts=0, train_freq=(10,'step'), gradient_steps=20,
                    target_update_interval=20, batch_size=4096, learning_rate=1e-6, policy_kwargs=self.policy_kwargs, device=torch.device(1))
        model.set_logger(new_logger)
        self.callback = callback_list
        return model, callback_list
    
    def learn(self, model:my_dqn, callback_list, env, train_from_scratch=False):
        if train_from_scratch:
            model.learn(total_timesteps=5e6, log_interval=5000, progress_bar=True, callback=callback_list, reset_num_timesteps=True,)
        else:
            model.load_replay_buffer(self.latest_buffer_path)
            model = model.load(self.latest_ckpt_path, env=env, buffer_size=100000, learning_starts=0, train_freq=(100,'step'), gradient_steps=20,
                    target_update_interval=400, batch_size=4096, learning_rate=1e-6, device=torch.device(1))
            model.learn(total_timesteps=5e6, log_interval=5000, progress_bar=True, callback=callback_list, reset_num_timesteps=False,)

    def get_buffer_name(self, buffer_step = None):
        file_list = os.listdir(self.base_path)
        buffer_file = [file_name for file_name in file_list if file_name.split('_')[2] == 'replay']
        if len(buffer_file) == 0:
            return None, None
        if buffer_step is None:
            # latest
            sorted_buffer_file = sorted(buffer_file, key=lambda x: int(x.split('_')[4]), reverse=True)
        else:
            # select
            sorted_buffer_file = [file_name for file_name in buffer_file if int(file_name.split('_')[4]) == buffer_step]
            assert len(sorted_buffer_file) == 1
        return sorted_buffer_file[0], int(sorted_buffer_file[0].split('_')[4])

    def get_ckpt_name(self, ckpt_step = None):
        file_list = os.listdir(self.base_path)
        ckpt_file = [file_name for file_name in file_list if file_name.split('_')[2].isdigit()]
        if len(ckpt_file) == 0:
            return None, None
        if ckpt_step is None:
            sorted_ckpt_file = sorted(ckpt_file, key=lambda x: int(x.split('_')[2]), reverse=True)
        else:
            sorted_ckpt_file = [file_name for file_name in ckpt_file if int(file_name.split('_')[2]) == ckpt_step]
            assert len(sorted_ckpt_file) == 1
        return sorted_ckpt_file[0], int(sorted_ckpt_file[0].split('_')[2])

    def evaluate(self, save_path = "./data/video/", episode_num = 5, time_out = 200):
        episode_step = 0
        time_step = 0
        total_reward = 0
        now = datetime.now()
        current_month = now.strftime("%m")
        current_day = now.strftime("%d")
        current_hour = now.strftime("%H")
        current_minute = now.strftime("%M")
        folder_name = f"{current_month}_{current_day}_{current_hour}_{current_minute}"
        folder_path = os.path.join(save_path, folder_name)
        print(f"setup folder path: {folder_path}")
        os.makedirs(folder_path)
        pb = tqdm()
        while episode_step < episode_num:
            info_args = {}
            action_prob = np.array([0,0,0,0], dtype=np.float64)
            pb.update(1)
            if self._last_obs is None:
                obs, info = self.env.reset()
                self._last_obs = copy.deepcopy(obs)
                info_args['topo'] = obs['map_topo']
                info_args['action_prob'] = action_prob
                arr_img = self.env.unwrapped.render(**info_args)
                img_name = f"ep_{episode_step}_timestep_{time_step}.jpg"
                cv2.imwrite(os.path.join(folder_path, img_name), arr_img)
                time_step += 1

            action, state = model.predict(observation=self._last_obs, deterministic=True)
            action_prob = model.policy.q_net.prob_values
            obs, reward, done, _, info = self.env.step(action=action)
            self._last_obs = copy.deepcopy(obs)
            total_reward += reward
            # draw
            # topo = obs['map_topo']
            info_args['topo'] = obs['map_topo']
            info_args['action_prob'] = action_prob
            arr_img = self.env.unwrapped.render(**info_args)
            img_name = f"ep_{episode_step}_timestep_{time_step}.jpg"
            cv2.imwrite(os.path.join(folder_path, img_name), arr_img)
            time_step += 1

            if done:
                episode_step += 1
                obs, info = self.env.reset()
                print(f"length: {time_step}, total_reward: {total_reward}")
                total_reward = 0
                time_step = 0
                # topo = obs['map_topo']
                info_args['topo'] = obs['map_topo']
                info_args['action_prob'] = np.array([0,0,0,0], dtype=np.float64)
                arr_img = self.env.unwrapped.render(**info_args)
                img_name = f"ep_{episode_step}_timestep_{time_step}.jpg"
                cv2.imwrite(os.path.join(folder_path, img_name), arr_img)
                time_step += 1
        pb.close()



if __name__ == "__main__":
    policy_kwargs = {
        "features_extractor_kwargs": {"net_arch":[8,32,16], },
        "net_arch": [64, 64, 16],
    }
    wandb_kwargs = {
        'wb_project': "terminal_ev_history",
        'wb_name': None,
        'wb_notes': None, 
        'wb_tags': None,
    }
    train = True
    robot = myrobot(base_path='./data/with_history/', buffer_step=None, ckpt_step=None, policy_kwargs=policy_kwargs, wandb_kwargs=wandb_kwargs)
    env = robot.set_env()
    
    model, callback_list = robot.set_model(env=env, wandb_flag=train)
    if train:
        robot.learn(model=model, callback_list=callback_list, env=env, train_from_scratch=False)
    else:
        robot.evaluate()
