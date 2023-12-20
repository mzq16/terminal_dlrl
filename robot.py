import stable_baselines3 as sb3
from stable_baselines3 import DQN
from dqn.my_dqn import my_dqn
from dqn.my_policy import myPolicy, myQNetwork
import gymnasium as gym
import gym_ENV
import time
import pygame
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv
import cv2


def train_CartPole():
    env = gym.make("CartPole-v1", render_mode="human")

    model = DQN("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000, log_interval=4)
    model.save("dqn_cartpole")

    del model # remove to demonstrate saving and loading

    model = DQN.load("dqn_cartpole")

    obs, info = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()

def test_multi_env():
    env = gym.make("CartPole-v1", render_mode="human")
    def make_env(env_id, rank, seed=0):
        def _init():
            env = gym.make(env_id)
            env.seed(seed + rank)
            return env
        return _init
    env_id = 'CartPole-v1'
    num_envs = 4
    a = {}
    env = SubprocVecEnv([make_env(env_id, i) for i in range(num_envs)], **a)
    

def should_quit():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return 1
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                return 1
            elif event.key == pygame.K_c:
                return 2
    return 0

def use_my_dqn():
    env_base = gym.make('Terminal_Env-v0', num_vehicle = 10, map_size = [1200, 700], 
                    render_mode = 'rgb_array', seed = 24, text_width = 400,
                    model_arg = {'input_dim': 2, 'hidden_dim': 64, 'output_dim': 2, 'num_layers': 2,},
                 )
    model = DQN(myPolicy, env_base, verbose=1)

def test():
    '''
    test: start_loc=[7, 1] des=[5,2]
    action:
    0: np.array([0, 0]), 
    1: np.array([0, 1]),
    2: np.array([0, -1]),
    3: np.array([-1, 0]),
    4: np.array([1, 0]),
    0:stop, 1:up, 2:down, 3:left, 4:right
    '''
    env_base = gym.make('gym_ENV/Terminal_Env-v0', num_vehicle = 10, map_size = [1200, 700], 
                    render_mode = 'rgb_array', seed = 24, text_width = 400,
                    model_arg = {'input_dim': 2, 'hidden_dim': 64, 'output_dim': 2, 'num_layers': 2,},
                    )
    obs, info = env_base.reset(seed=54)
    done = False
    while not done:
        obs, r, done, _, info = env_base.step(0)
        '''
        train
        obs, r -> network
        network -> action
        ege_V action -> next loc, according to road
        e.g. (0,0)->(1,0),(0,1)   road_id road information 
        function, feasible node: set[(0,0)] = {((1,0),(0,1))} distance
        road_id <--> xy 
        
        '''
        #print(i)
        if done:
            break
        env_base.render()
        flag = should_quit()
        if flag:
            break
    env_base.close()

def test_render():
    env = gym.make('Terminal_Env-v0', num_vehicle = 10, map_size = [1200, 700], 
               render_mode = 'rgb_array', seed = 24, text_width = 400,
               model_arg = {'input_dim': 2, 'hidden_dim': 64, 'output_dim': 2, 'num_layers': 2,},
            )
    obs, info = env.reset()
    env.get_wrapper_attr('ev_handle').start_id = 0
    env.get_wrapper_attr('ev_handle').target_id = 2
    arr = env.render()
    cv2.imwrite('dd.jpg',arr)

def main():
    '''
    train
    action:
    0: np.array([0, 0]), 
    1: np.array([0, 1]),
    2: np.array([0, -1]),
    3: np.array([-1, 0]),
    4: np.array([1, 0]),
    0:stop, 1:up, 2:down, 3:left, 4:right
    '''
    env_base = gym.make('Terminal_Env-v0', num_vehicle = 10, map_size = [1200, 700], 
                    render_mode = 'rgb_array', seed = 24, text_width = 400,
                    model_arg = {'input_dim': 2, 'hidden_dim': 64, 'output_dim': 2, 'num_layers': 2,},
                    )
    model = DQN("MultiInputPolicy", env_base, verbose=1)

        
        

if __name__ == '__main__':
    #test()
    #test_multi_env()
    test_render()