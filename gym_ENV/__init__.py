from gymnasium.envs.registration import register
# from gym.envs.registration import register
from . import envs

register(
     id="Terminal_Env-v0",
     entry_point="gym_ENV.envs:Terminal_Env",
     max_episode_steps=300,
)

'''
the following is old version
register(
    id='Terminal_Env-v0',
    entry_point='gym_ENV.envs:Terminal_Env',
) 
'''
