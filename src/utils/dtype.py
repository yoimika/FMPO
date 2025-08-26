import torch
import numpy 

def numpy2torch(array: numpy.ndarray) -> torch.Tensor:
    if isinstance(array, torch.Tensor):
        return array.float()
    return torch.from_numpy(array).float()

def torch2numpy(tensor: torch.Tensor) -> numpy.ndarray:
    if isinstance(tensor, numpy.ndarray):
        return tensor
    return tensor.detach().cpu().numpy()

def list2torch(item: list):
    return numpy2torch(numpy.array(item))