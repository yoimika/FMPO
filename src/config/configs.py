import torch
from torch.utils.data import DataLoader
from dataclasses import dataclass, field
from env import *
from utils import *

@dataclass
class BaseTrainConfig:
    learning_rate: float = 1e-4
    batch_size: int = 256
    epoches: int = 256

    optim_name: str = "Adam"

    save_dir: str = './save/'
    save_file_name: str = 'model.pth'

    def get_optimizer(self, params):
        if self.optim_name == "Adam":
            return torch.optim.Adam(params, lr=self.learning_rate)
        elif self.optim_name == "AdamW":
            return torch.optim.AdamW(params, lr=self.learning_rate)
        elif self.optim_name == "SGD":
            return torch.optim.SGD(params, lr=self.learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {self.optim_name}")

@dataclass
class FlowTrainConfig(BaseTrainConfig):
    teacher_dataset_fp: str = './save/sac-teacher.pkl' # Teacher dataset path for imitation learning.

    save_dir: str = './save/'
    save_file_name: str = 'flow-imitation.pth'

    def get_teacher_dataloader(self):
        def collate_fn(batch):
            ret_data = {}
            ret_data['observation'] = from_numpy( np.stack(
                [gym_robotics_observation_concat(item['observation']) for item in batch], axis=0
            ) )
            ret_data['action'] = from_numpy( np.stack([item['action'] for item in batch], axis=0) )
            return ret_data

        data = load_dataset(self.teacher_dataset_fp, flat=True)
        dataloader = DataLoader(data, batch_size=self.batch_size, shuffle=True, drop_last=True, collate_fn=collate_fn)
        return dataloader
    
@dataclass
class RLTrainConfig(BaseTrainConfig):
    algorithm: str = "PPO"

    episode_length: int = 10000

    base_model_file_name: str = "flow-imitation.pth"

    env_name: str = "FetchPickAndPlace-v4"
    env_wrapper: list[Wrapper] = field(default_factory=lambda: [
        RoboticsWrapper,
    ])
    vectorize_env: bool = True
    num_envs: int = 32

    clip_epsilon = 0.05
    average_loss_before_exp: bool = True

    save_interval: int = 10
    save_idx: bool = False
    load_idx: int = None

    def get_env(self, device: torch.device = torch.device('cpu')):
        if self.vectorize_env:
            env = create_robotics_env(self.env_name, vec=self.vectorize_env, num_env=self.num_envs, device=device)
        else:
            env = create_robotics_env(self.env_name, vec=self.vectorize_env, device=device)
        return env