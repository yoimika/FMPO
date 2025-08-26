import numpy as np
import torch
from utils.dtype import torch2numpy, list2torch

class Buffer:
    def __init__(self):
        self.episodes = []
        self.clear()

    def push(self, dict):
        self.episodes.append( dict )

    def clear(self):
        self.episodes = []
    
    def compute_rewards_to_go(self, gamma: float):
        for episode in self.episodes:
            episode["rewards_to_go"] = []
            rewards = episode["rewards"]
            rew = 0
            for r in reversed(rewards):
                rew = r + gamma * rew
                episode["rewards_to_go"].insert(0, rew)
            assert len(episode["rewards_to_go"]) == len(episode["rewards"])
            episode["rewards_to_go"] = list2torch(episode["rewards_to_go"])

    def get_all_use_key(self, key: str):
        return torch.cat([episode[key] for episode in self.episodes], dim=0)