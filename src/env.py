from copy import deepcopy as dc
import gymnasium as gym
import gymnasium_robotics
import cv2

from flow import Flow
gym.register_envs(gymnasium_robotics)
import numpy as np
import torch
from tqdm import tqdm
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
        return from_numpy(obs, self.device), from_dict(info, self.device)

    def step(self, action):
        _action = from_tensor(action)
        obs, reward, terminated, truncated, info = self.env.step(_action)
        return from_numpy(obs, self.device), from_numpy(reward, self.device), from_numpy(terminated, self.device), from_numpy(truncated, self.device), from_dict(info, self.device)

class VectorTorchWrapper(gym.vector.VectorWrapper):
    def __init__(self, env: gym.vector.VectorEnv, device: torch.device = torch.device('cpu')):
        super(VectorTorchWrapper, self).__init__(env)
        self.device = device
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return from_numpy(obs, self.device), from_dict(info, self.device)

    def step(self, action):
        _action = from_tensor(action)
        obs, reward, terminated, truncated, info = self.env.step(_action)
        return from_numpy(obs, self.device), from_numpy(reward, self.device), from_numpy(terminated, self.device), from_numpy(truncated, self.device), from_dict(info, self.device)


def create_robotics_env(env_name: str, vec: bool, render: bool = False, mp4: bool = False, num_env: int = 1, device: torch.device = torch.device('cpu')):
    """
    vec -> vec
    render and mp4 -> record mp4
    render and not mp4 -> human render
    """
    if vec:
        env = gym.make_vec(env_name, num_envs=num_env, vectorization_mode='sync', wrappers=[
            lambda e: RoboticsWrapper(e),
        ])
        env = VectorTorchWrapper(env, device=device)
    else:
        if render:
            if mp4:
                env = gym.make(env_name, render_mode='rgb_array')
            else:
                env = gym.make(env_name, render_mode='human')
        else:
            env = gym.make(env_name)
        env = RoboticsWrapper(env)
        env = TorchWrapper(env, device=device)
    return env

def eval_robotics_env(env_name: str, flow: Flow, sample_nums: int, device: torch.device = torch.device('cpu'), render: bool = False):
    mp4 = False
    """
    mp4 and render -> record mp4
    not mp4 and render -> human render
    else -> no render
    """
    if mp4:
        assert render == True
    vec = True
    if render: 
        vec = False
    env = create_robotics_env(env_name, vec=vec, render=render, num_env=32, device=device, mp4=mp4)
    success_transitions_count = 0
    all_transitions_count = 0
    rews = []

    if mp4:
        obs, _ = env.reset()
        if mp4:
            frame = env.render()
            H, W, _ = frame.shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter('./save/Demo_rgb.mp4', fourcc, 30.0, (W, H))
    for idx in tqdm( range(sample_nums) ):
        obs, _ = env.reset()
        done = np.array(0.0) if vec else False
        raw_traj = []
        while not (done.sum().item() if vec else done):
            if len(obs.shape) == 1:
                obs = obs.unsqueeze(0)
            # import pdb; pdb.set_trace()
            action, _ = flow.sample_action(obs)  # (1, action_dim)
            obs, rew, terminated, truncated, info = env.step(action.cpu().squeeze().numpy())
            done = (terminated + truncated) if vec else (terminated or truncated)
            success_transitions_count += info['is_success'].sum()
            all_transitions_count += len(obs)
            raw_traj.append(rew.cpu().numpy() if isinstance(rew, torch.Tensor) else rew)

            if mp4:
                frame = env.render()
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
        
        traj = np.stack(raw_traj).squeeze() + 1
        if True:
            # # Reward to go
            # for i in reversed(range(len(traj))):
            #     traj[i] = traj[i] + (traj[i+1] if i+1 < len(traj) else 0) * 0.95

            # Segmented Reward
            rew = 0
            for i in reversed(range(len(traj))):
                rew = traj[i] + rew * 0.95 * (traj[i] == 0)
                traj[i] = rew

            # stat_traj = traj[-5:] - 1
            # sum_stat_rew = sum(stat_traj)
            # stat_rew = dc(sum_stat_rew)
            # stat_rew[sum_stat_rew == 0] = 1
            # stat_rew[sum_stat_rew != 0] = 0
            # for i in reversed(range(len(traj))):
            #     traj[i] = stat_rew
            # import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()
        rews.extend( traj )
    if mp4:
        out.release()
    stacked_rews = (np.stack(rews))
    success_ratio = success_transitions_count / all_transitions_count
    print(f"Eval robotics env success ratio: {success_ratio:.4f}")
    print(f"Average Rewards: {stacked_rews.mean():.4f} +/- {stacked_rews.std():.4f}")


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

    