from env import EnvConfig
from flow import Flow, ActionInfo, Transition
from dataclasses import dataclass, field
import torch
import gymnasium as gym
from tqdm import tqdm
import numpy as np

def compute_return_to_go(trajectory: list[Transition], gamma=0.95):
    # Reward to go
    rew = 0
    for i in reversed(range(len(trajectory))):
        rew = trajectory[i].reward + gamma * rew
        trajectory[i].reward_to_go = rew


def rollout(flow: Flow, env: gym.Env, iter: int, pbar: tqdm = None, env_config: EnvConfig = None):
    trajectories = []
    for i in range(iter):
        trajectory = []
        obs, _ = env.reset()
        done = np.array([0])
        while not done.sum():
            action, action_info = flow.sample_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated + truncated
            # Trick: Reward + 1 for failure 0, success 1
            trajectory.append(Transition(
                obs=obs, 
                next_obs=next_obs, 
                action=action, 
                reward=reward, 
                done=done, 
                action_info=action_info,
                reward_to_go=torch.zeros_like(reward),
            ))
            obs = next_obs

        if pbar:
            pbar.set_description(f"Rollout {i+1}/{iter}")
        compute_return_to_go(trajectory, env_config.env_gamma)
        # import pdb;   pdb.set_trace()
        trajectories.extend(trajectory)
    return RolloutState(trajectories)

class RolloutState(Transition):
    def __init__(self, transitions: list[Transition]):
        def stack_tensor(attr):
            return torch.stack(attr, dim=0)
        self.obs = stack_tensor([t.obs for t in transitions])
        self.next_obs = stack_tensor([t.next_obs for t in transitions])
        self.action = stack_tensor([t.action for t in transitions])
        self.reward = stack_tensor([t.reward for t in transitions])
        self.reward_to_go = stack_tensor([t.reward_to_go for t in transitions])
        self.done = stack_tensor([t.done for t in transitions])
        self.action_info = ActionInfo(
            cfm_loss=stack_tensor([t.action_info.cfm_loss for t in transitions]),
            t=stack_tensor([t.action_info.t for t in transitions]),
            x1=stack_tensor([t.action_info.x1 for t in transitions])
        )
        
        self.post_reward_handle()
    
    def prepare_batches(self, batch_size):
        T, B, _ = self.obs.shape
        # print(T, B)
        assert T * B % batch_size == 0
        length = T * B // batch_size

        def _prepare_single_batches(item):
            suffix = item.shape[2:]
            return item.view(length, batch_size, *suffix)

        obs = _prepare_single_batches(self.obs)
        next_obs = _prepare_single_batches(self.next_obs)
        action = _prepare_single_batches(self.action)
        reward = _prepare_single_batches(self.reward)
        rew_to_go = _prepare_single_batches(self.reward_to_go)
        done = _prepare_single_batches(self.done)

        cfm_loss = _prepare_single_batches(self.action_info.cfm_loss)
        t = _prepare_single_batches(self.action_info.t)
        x1 = _prepare_single_batches(self.action_info.x1)

        return [
            Transition(
                obs=obs[i], 
                next_obs=next_obs[i], 
                action=action[i], 
                reward=reward[i], 
                reward_to_go=rew_to_go[i],
                done=done[i], 
                action_info=ActionInfo(
                    cfm_loss=cfm_loss[i], 
                    t=t[i], 
                    x1=x1[i]
                )
            ) for i in range(length)
        ]
    
    def post_reward_handle(self):
        """对 reward (T, B) 做一些处理"""

        # import pdb; pdb.set_trace()
        # list 保证 trajectory 的时序性
        # # 分段 Reward
        # rew = 0
        # for i in reversed(range(len(trajectory))):
        #     rew = trajectory[i].reward + gamma * rew * (trajectory[i].reward == 0)
        #     trajectory[i].reward = rew
        
        # 标准化每一个 reward 相当于是 batch normalization
        mean, std = self.reward.mean(dim=-1, keepdim=True), self.reward.std(dim=-1, keepdim=True) + 1e-5
        self.reward = (self.reward - mean) / std