import gymnasium as gym
import numpy as np
from typing import Dict, Any, Tuple

class FlattenObservationWrapper(gym.ObservationWrapper):
    """将dict observation扁平化，只保留'observation'字段，其他字段移到info"""
    
    def __init__(self, env, main_obs_key='observation'):
        super().__init__(env)
        self.main_obs_key = main_obs_key
        
        # 获取主要观测空间
        original_obs_space = env.observation_space
        if isinstance(original_obs_space, gym.spaces.Dict):
            if main_obs_key in original_obs_space.spaces:
                # 新的观测空间只包含主要观测
                self.observation_space = original_obs_space.spaces[main_obs_key]
                # 记录其他字段的空间信息
                self.other_obs_keys = [k for k in original_obs_space.spaces.keys() if k != main_obs_key]
            else:
                raise ValueError(f"Key '{main_obs_key}' not found in observation space")
        else:
            raise ValueError("Environment observation space must be Dict type")
    
    def observation(self, obs_dict):
        """只返回主要观测"""
        return obs_dict[self.main_obs_key]
    
    def step(self, action):
        obs_dict, reward, terminated, truncated, info = self.env.step(action)
        
        # 将其他观测字段添加到info中
        for key in self.other_obs_keys:
            if key in obs_dict:
                info[f'obs_{key}'] = obs_dict[key]
        
        # 返回扁平化的观测
        flattened_obs = self.observation(obs_dict)
        return flattened_obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        obs_dict, info = self.env.reset(**kwargs)
        
        # 将其他观测字段添加到info中
        for key in self.other_obs_keys:
            if key in obs_dict:
                info[f'obs_{key}'] = obs_dict[key]
        
        # 返回扁平化的观测
        flattened_obs = self.observation(obs_dict)
        return flattened_obs, info
