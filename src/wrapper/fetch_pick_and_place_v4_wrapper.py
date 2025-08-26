import gymnasium as gym
from gymnasium import Wrapper
import numpy as np


class FetchPickAndPlaceV4Wrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(9,), dtype=np.float32)
    
    def observation(self, obs):
        ret_obs = np.concatenate([obs['observation'][:6], obs['desired_goal']], axis=0)
        return ret_obs
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        ret_obs = self.observation(obs)
        relative_block2gripper = ret_obs[6:9]
        ret_rew = reward - np.linalg.norm(relative_block2gripper)
        return ret_obs, ret_rew, terminated, truncated, info
    
    def reset(self):
        obs, info = self.env.reset()
        ret_obs = self.observation(obs)
        return ret_obs, info