from gymnasium import Wrapper
import yaml
import pickle
import torch
import numpy as np
from functools import reduce

def print_green(str: str):
    print(f"\033[92m{str}\033[0m")

def from_numpy(obj):
    if isinstance(obj, np.ndarray):
        return torch.from_numpy(obj).float()
    return obj

def from_tensor(obj):
    if isinstance(obj, torch.Tensor):
        return obj.cpu().numpy()
    return obj

def from_dict(obj: dict):
    return {k: from_numpy(v) for k, v in obj.items()}

def gym_robotics_observation_concat(obs):
    _obs = obs['observation']
    _goal = obs['desired_goal']
    ret_obs = np.concatenate([_obs, _goal], axis=-1)
    return ret_obs

def load_dataset(dataset_fp: str, flat: bool=True):
    with open(dataset_fp, 'rb') as f:
        data = pickle.load(f)
    if not flat:
        return data
    ret_data = []
    for traj in data:
        for transition in traj:
            ret_data.append(transition)
    return ret_data

def load_yaml(fp):
    with open(fp, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return cfg

def overwrite_object(obj: object, data: dict):
    for k, v in data.items():
        if hasattr(obj, k):
            setattr(obj, k, v)
    return obj

def env_wrapper(env, wrappers: list[tuple[Wrapper, dict]]):
    return reduce(
        lambda _env, wrapper_args: wrapper_args[0](_env, **wrapper_args[1]),
        wrappers,
        env
    )

if __name__ == '__main__':
    fp = './config/flow.yaml'
    from flow import FlowConfig
    flow_config = FlowConfig()
    cfg = load_yaml(fp)
    print(cfg)
    overwrite_object(flow_config, cfg['flow'])

    print(flow_config)