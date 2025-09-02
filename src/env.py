import gymnasium as gym
import gymnasium_robotics
gym.register_envs(gymnasium_robotics)
import numpy as np
import torch
from utils import from_numpy, from_tensor, from_dict, gym_robotics_observation_concat

class RoboticsWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super(RoboticsWrapper, self).__init__(env)
        _observation = env.observation_space['observation']
        _desired_goal = env.observation_space['desired_goal']

        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(_observation.shape[0] + _desired_goal.shape[0],), dtype=np.float32)
        self.action_space = env.action_space

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = gym_robotics_observation_concat(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return gym_robotics_observation_concat(obs), reward, terminated, truncated, info
    

class TorchWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, device: torch.device = torch.device('cpu')):
        super(TorchWrapper, self).__init__(env)
        self.device = device

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return from_numpy(obs), from_dict(info)

    def step(self, action):
        _action = from_tensor(action)
        obs, reward, terminated, truncated, info = self.env.step(_action)
        return from_numpy(obs), from_numpy(reward), from_numpy(terminated), from_numpy(truncated), from_dict(info)
    
class VectorTorchWrapper(gym.vector.VectorWrapper):
    def __init__(self, env: gym.vector.VectorEnv, device: torch.device = torch.device('cpu')):
        super(VectorTorchWrapper, self).__init__(env)
        self.device = device
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return from_numpy(obs), from_dict(info)

    def step(self, action):
        _action = from_tensor(action)
        obs, reward, terminated, truncated, info = self.env.step(_action)
        return from_numpy(obs), from_numpy(reward), from_numpy(terminated), from_numpy(truncated), from_dict(info)


def create_robotics_env(env_name: str, vec: bool, render: bool = False, num_env: int = 1):
    if vec:
        env = gym.make_vec(env_name, num_envs=num_env, vectorization_mode='sync', wrappers=[
            lambda e: RoboticsWrapper(e),
        ])
        env = VectorTorchWrapper(env)
    else:
        if render:
            env = gym.make(env_name, render_mode='human')
        else:
            env = gym.make(env_name)
        env = RoboticsWrapper(env)
        env = TorchWrapper(env)
    return env

if __name__ == '__main__':
    env_name = 'FetchPickAndPlace-v4'
    # env = gym.make(env_name)
    env = gym.make_vec(env_name, num_envs=10, vectorization_mode='sync', wrappers=[
        lambda e: RoboticsWrapper(e)
    ])
    print(type(env))
    # env = RoboticsWrapper(env)
    env = VectorTorchWrapper(env)

    obs, info = env.reset()
    print(len(obs), type(obs), info)
    next_obs, reward, term, trun, next_info = env.step(env.action_space.sample())
    next_obs, reward, term, trun, next_info = env.step(env.action_space.sample())
    print(type(reward))
    print(next_info)
    env.close()

    