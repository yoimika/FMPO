import pickle
import torch
import numpy as np

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
    