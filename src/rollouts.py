from dataclasses import dataclass
import torch
import torch.nn
import numpy as np
from typing import Any

@dataclass
class Bags:
    x0: torch.Tensor
    x1: torch.Tensor
    t: torch.Tensor
    initial_cfm_loss: torch.Tensor

    def prepare_minibatch(self, mini_bs: int):
        total_bs = self.x0.shape[0]
        assert total_bs % mini_bs == 0

        out = []
        indices = torch.randperm(total_bs)
        for i in range(0, total_bs, total_bs // mini_bs):
            sub_indices = indices[i:i+mini_bs]
            sub_bags = Bags(
                x0 = self.x0[sub_indices],
                x1 = self.x1[sub_indices],
                t = self.t[sub_indices],
                initial_cfm_loss=self.initial_cfm_loss[sub_indices]
            )
            out.append(sub_bags)
        return out
