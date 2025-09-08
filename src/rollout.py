from flow import Flow, ActionInfo, Transition
from dataclasses import dataclass, field
import torch
import gymnasium as gym
from tqdm import tqdm

def compute_return(trajectory: list[Transition], gamma=0.95):
    # # Nothing to do 
    # return 

    # Reward to go
    rew = 0
    for i in reversed(range(len(trajectory))):
        rew = trajectory[i].reward + gamma * rew
        trajectory[i].reward_to_go = rew

    # 分段 Reward
    rew = 0
    for i in reversed(range(len(trajectory))):
        rew = trajectory[i].reward + gamma * rew * (trajectory[i].reward == 0)
        trajectory[i].reward = rew
    
    # # 同一 Reward
    # rew = 0
    # statistic_traj = trajectory[:-5]
    # statistic_rewd = sum([t.reward - 1 for t in statistic_traj])
    # stat_rew = dc(statistic_rewd)
    # stat_rew[statistic_rewd == 0] = 1
    # stat_rew[statistic_rewd != 0] = 0
    # for i in reversed(range(len(trajectory))):
    #     trajectory[i].reward = dc(stat_rew)

def rollout(flow: Flow, env: gym.Env, iter: int, pbar: tqdm = None):
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
            trajectory.append(Transition(obs, next_obs, action, reward+1, done, action_info))
            obs = next_obs

        if pbar:
            pbar.set_description(f"Rollout {i+1}/{iter}")
        compute_reward_to_go = True
        if compute_reward_to_go:
            compute_return(trajectory)
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