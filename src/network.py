import torch
import torch.nn as nn
import numpy as np

def make_layers(layers_info: list[int]) -> nn.Sequential:
    layers = []
    layers_info = list(zip(layers_info[:-1], layers_info[1:]))
    for i in range(len(layers_info)-1):
        in_dim, out_dim = layers_info[i]
        layers.append(nn.Linear(in_dim, out_dim))
        layers.append(nn.SiLU(inplace=True))
    in_dim, out_dim = layers_info[-1]
    layers.append(nn.Linear(in_dim, out_dim))
    return nn.Sequential(*layers)

class MLP(nn.Module):
    def __init__(self, layers_info:list[int]):
        super().__init__()
        self.layers = make_layers(layers_info)
    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x, dtype=torch.float32)
        return self.layers(x)
    