from env import EnvConfig
from flow import Flow, ActionInfo, Transition
from dataclasses import dataclass, field
import torch
import gymnasium as gym
from tqdm import tqdm
import numpy as np
from copy import deepcopy as dc

def compute_return(trajectory: list[Transition], gamma=0.95):
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
        compute_return(trajectory, env_config.env_gamma)
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
        
        # # 标准化每一个 reward 相当于是 batch normalization
        # mean, std = self.reward.mean(dim=-1, keepdim=True), self.reward.std(dim=-1, keepdim=True) + 1e-5
        # # import pdb; pdb.set_trace()
        # self.reward = (self.reward - mean) / std
        # # 不行！因为初始状态不同，你不能保证在同一个时间段的 state 有同等地位。

        # # 用 return 代替 trajectory 的所有 transition 的 reward
        # self.reward = torch.expand_copy(self.reward_to_go[0:1, ...], self.reward.shape)
        # mean, std = self.reward.mean(), self.reward.std() + 1e-5
        # self.reward = (self.reward - mean) / std
        # # import pdb; pdb.set_trace()
        # # 这样做的好处是，所有 transition 的 reward 都是同等地位的。
        # # 但是也不行！因为环境初始化不同，会导致即使是最优的策略，不同的 trajectory 就是会不一样

        # Do Nothing （也许这样是最好的🥲）
        # 用 critic 就可以完全规避这些情况。因为我们需要衡量的是 action 的价值
        # 所以希望衡量的标准只和 action 有关，而和 state 无关
        return

