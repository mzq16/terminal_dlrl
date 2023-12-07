from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from pathlib import Path
import wandb
import time
import numpy as np

class WandbCallback(BaseCallback):
    def __init__(self, cfg, vec_env, video_path, ckpt_dir, buffer_dir):
        super(WandbCallback, self).__init__(verbose=1)

        save_dir = Path.cwd()
        print(save_dir)
        self._video_path = Path(video_path)
        self._video_path.mkdir(parents=True, exist_ok=True)
        self._ckpt_dir = Path(ckpt_dir)
        self._ckpt_dir.mkdir(parents=True, exist_ok=True)
        self._buffer_dir = Path(buffer_dir)
        self._buffer_dir.mkdir(parents=True, exist_ok=True)

        
        wandb.init(project=cfg['wb_project'], name=cfg['wb_name'], notes=cfg['wb_notes'], tags=cfg['wb_tags'])
        self.vec_env = vec_env

        self._eval_step = int(1e4)
        #self._buffer_step = int(1e4)
        self._save_step = int(1e3)
        self._save_buffer_step = int(5e3)

    def _init_callback(self):
        self.n_epoch = 0
        self._learning_starts = self.model.learning_starts
        self._last_time_buffer = self.model.num_timesteps
        self._last_time_eval = self.model.num_timesteps
        self._last_time_save = self.model.num_timesteps
        self._last_time_save_buffer = self.model.num_timesteps

    def _on_step(self) -> bool:
        # 被on_step调用，on_step在BaseCallback中，一般返回true，被rollout每一步调用

        return True

    def _on_training_start(self) -> None:
        # 父类有一个不是隐的，会调用这个，并且会加入local和global
        self.train_start_time = time.time()
        print('training start')
        
    def _on_rollout_start(self):
        # 同上，父类有一个非隐调用
        self.rollout_start_time = time.time()
        print('rollout start')

    def _on_training_end(self) -> None:
        # 这个地方是learn的最后
        # train time 其实是learn time，包括rollout的时间
        train_time = time.time() - self.train_start_time
        time_elapsed = time.time() - self.model.start_time
       
        wandb.log({
            'train/learning_rate': self.model.learning_rate,
            'train/ent_coef': self.locals['local_ent_coef'],
        })
        
        wandb.log({
            'time/fps': (self.model.num_timesteps-self.model._num_timesteps_at_start) / time_elapsed,
            'time/total_time_elapsed': time_elapsed,
            'time/total_learn_time': train_time,
            'time/rollout': self.model.t_rollout
        }, step=self.model.num_timesteps)

        # evaluate and save checkpoint
    

    def _on_rollout_end(self):
        rollout_time = time.time() - self.rollout_start_time
        t0 = time.time()
        wandb.log({
            'rollout/tmp_z': self.locals['tmp_z'],
            'rollout/life_step': self.model._life_step,
        }, step=self.model.num_timesteps)

def init_callback_list(save_freq=10, save_path='data/checkpoint/', verbose=2):
    '''
    verbose: 2就是什么信息都打印，0就都不打印，没1什么事，之后可以改
    '''
    callback_save = CheckpointCallback(save_freq=save_freq, save_path=save_path, verbose=verbose)
    callback_wandb = WandbCallback()

    # 这个callback list 看了看源码应该是可以嵌套的，因为callback list也是继承了BaseCallback，
    # 嵌套就是[callback1, [callback2,callback0]]，
    # callback3 = [callback2,callback0]
    # 之后也是一一调用callback1._on_step，然后callback3._on_step-------> callback2._on_step, callback0._on_step
    # 没区别，不嵌套了，后面还有一个tqdm的callback，那个是把rollout和train放一块了，如果任务比较简单就和成一起吧
    # callback_list = CallbackList([callback_save, callback_wandb])

    callback_list = [callback_save, callback_wandb]
    return callback_list