from typing import Literal
import torch
import torch.nn as nn
from network import MLP, make_layers
import numpy as np
from dataclasses import dataclass
from rollouts import Bags


@dataclass
class FlowConfig:
    deterministic: bool = False
    discretize_t: bool = True
    ratio_clip: bool = True
    normalize_rewards: bool = False

    clip_epsilon: float = 0.05


    output_mode: Literal["u", "u_but_supervised_as_eps"] = "u_but_supervised_as_eps"

    flow_vel_sacle: float = 0.25
    sde_sigma: float = 0.0
    feather_std: float = 0.0

    input_dim: int = 1
    output_dim: int = 1
    time_embed_dim: int = 1

    time_steps: int = 10
    batch_size: int = 4096

    steps_per_training_steps: int = 16

    reward_scale: float = 1.0

class Flow(nn.Module):
    def __init__(self, config: FlowConfig):
        super().__init__()
        self.config = config
        self.time_embeder = MLP(layers_info=[1, 8, self.config.time_embed_dim])
        self.v_predictor = MLP(layers_info=[self.config.input_dim+self.config.output_dim, 64, 64, self.config.output_dim])

        self.normal_dist = torch.distributions.Normal(0, 1)
    
    def get_scheduler(self, N:int = None):
        N = self.config.batch_size if N is None else N
        t = torch.linspace(0, 1, steps=self.config.time_steps+1).expand((N, self.config.time_steps+1))
        return t[..., :-1], t[..., 1:]

    def pred(self, x, t):
        time_embed = self.time_embeder(t)
        x = torch.concat([x, time_embed], dim=-1)
        v = self.v_predictor(x) * self.config.flow_vel_sacle
        return v

    def euler_step(self, xt, start_t, end_t):
        dt = end_t - start_t

        v = self.pred(xt, start_t)

        next_x = xt + v * dt + self.config.sde_sigma * self.normal_dist.sample(v.shape)

        return next_x

    def sample_actions(self, sample_num: int = None):
        with torch.no_grad():
            sample_num = self.config.batch_size if sample_num is None else sample_num
            scheduler = self.get_scheduler(N=sample_num)
            x = self.normal_dist.sample((sample_num, self.config.input_dim))
            for idx in range(self.config.time_steps):
                st = scheduler[0][..., idx:idx+1]
                et = scheduler[1][..., idx:idx+1]
                x = self.euler_step(x, st, et)
            
            if not self.config.deterministic:
                pertub = self.normal_dist.sample((sample_num, self.config.input_dim))
                x = x + pertub

            eps = self.normal_dist.sample((sample_num, self.config.input_dim))

            if self.config.discretize_t:
                st, et = self.get_scheduler(N=sample_num)
                t = torch.concat([st, et[..., -1:]], dim=-1)
                idx = torch.randperm(self.config.time_steps+1)[:self.config.input_dim]
                t = t[..., idx]
            else:
                t = torch.rand((sample_num, self.config.input_dim))
            
            initial_cfm_loss = self.compute_cfm_loss(x, eps, t)
        return x, Bags(x, eps, t, initial_cfm_loss)

    def compute_cfm_loss(self, x0, eps, t):
        xt = (1 - t) * eps + t * x0
        v_pred = self.pred(xt, t)

        if self.config.output_mode == "u":
            v_gt = x0 - eps
            out = (v_pred - v_gt) ** 2
        elif self.config.output_mode == "u_but_supervised_as_eps":
            x1_pred = xt - t * v_pred
            # x1_pred = x0_pred + v_pred
            out = (eps - x1_pred) ** 2
        else:
            raise "LASKJFLJSAL"
        return out
    
    def compute_fpo_loss(self, bags: Bags, rewards):
        if self.config.normalize_rewards:
            rewards_norm = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        else:
            rewards_norm = rewards / (rewards.std() + 1e-8)
        rewards_norm = rewards_norm * self.config.reward_scale

        x0 = bags.x0
        eps = bags.x1
        t = bags.t
        initial_cfm_loss = bags.initial_cfm_loss 

        cfm_loss = self.compute_cfm_loss(x0, eps, t)

        if self.config.ratio_clip:
            ratio = torch.exp(
                torch.clip(initial_cfm_loss - cfm_loss, -3.0, 3.0)
            )
        else:
            ratio = torch.exp(initial_cfm_loss - cfm_loss)

        surr1 = ratio * rewards_norm
        surr2 = torch.clip(ratio, 1-self.config.clip_epsilon, 1+self.config.clip_epsilon) * rewards_norm

        policy_loss = -torch.min(surr1, surr2).mean()

        return policy_loss

    
    def rl_training_step(self, bags: Bags, reward_model, optim):
        mini_bs = self.config.batch_size // self.config.steps_per_training_steps
        batch_bags = bags.prepare_minibatch(mini_bs)
        for idx in range(self.config.steps_per_training_steps):
            mini_bags = batch_bags[idx]
            rewards = reward_model(mini_bags.x0)
            loss = self.compute_fpo_loss(mini_bags, rewards)

            optim.zero_grad()
            loss.backward()
            optim.step()
        return loss.item()
        

    def fm_training_step(self, x0):
        eps = self.normal_dist.sample((self.config.batch_size, self.config.input_dim))
        if self.config.discretize_t:
            st, et = self.get_scheduler()
            t = torch.concat([st, et[..., -1:]], dim=-1)
            idx = torch.randperm(self.config.time_steps+1)[:self.config.input_dim]
            t = t[..., idx]
        else:
            t = torch.rand((self.config.batch_size, self.config.input_dim))
        loss = self.compute_cfm_loss(x0, eps, t).mean()
        return loss


    

