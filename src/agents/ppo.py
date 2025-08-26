from network import MLP
from agents.base_agent import BaseAgent
from dataclasses import dataclass
from utils.dtype import numpy2torch
import numpy as np
import torch
import torch.nn as nn 

@dataclass
class PPOConfig:
    actor_layers: list[int] = None
    critic_layers: list[int] = None

    sigma: float = 0.02


class PPO(BaseAgent, nn.Module):
    def __init__(self, config: PPOConfig):
        super().__init__()
        self.config = config
        self.actor = MLP(self.config.actor_layers)
        self.critic = MLP(self.config.critic_layers)

        cov_var = torch.full(size=(self.config.actor_layers[-1],), fill_value=self.config.sigma)
        self.cov = torch.diag(cov_var)
    
    def forward(self, x):
        return self.actor(x)
    

    def sample_action(self, obs):
        with torch.no_grad():
            if isinstance(obs, np.ndarray):
                obs = numpy2torch(obs)
            mean = self.actor(obs)
            dist = torch.distributions.MultivariateNormal(mean, self.cov)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            return action.numpy(), log_prob.numpy()
    
    def get_value(self, obs):
        if isinstance(obs, np.ndarray):
            obs = numpy2torch(obs)
        return self.critic(obs).squeeze() # ****
    
    def get_log_prob(self, obs, act):
        mean = self.actor(obs)
        dist = torch.distributions.MultivariateNormal(mean, self.cov)
        log_prob = dist.log_prob(act)
        return log_prob
