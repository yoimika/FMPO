import torch
from torch.utils.data import DataLoader
from flow import Flow
from dataclasses import dataclass, field
from utils import load_dataset, gym_robotics_observation_concat
from tqdm import tqdm
import os

@dataclass
class FlowTrainConfig:
    learning_rate: float = 1e-4
    batch_size: int = 256

    optim_name: str = "Adam"

    teacher_dataset_fp: str = './save/sac-teacher.pkl' # Teacher dataset path for imitation learning.

    save_dir: str = './save/'

    def get_optimizer(self, params):
        if self.optim_name == "Adam":
            return torch.optim.Adam(params, lr=self.learning_rate)
        elif self.optim_name == "SGD":
            return torch.optim.SGD(params, lr=self.learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {self.optim_name}")
    
    def get_teacher_dataloader(self):
        def collate_fn(batch):
            def func(data):
                data['observation'] = gym_robotics_observation_concat(data['observation'])
                return data
            return torch.vmap(func)(batch)
        data = load_dataset(self.teacher_dataset_fp, flat=True)
        dataloader = DataLoader(data, batch_size=self.batch_size, shuffle=True, drop_last=True, collate_fn=collate_fn)
        return dataloader

@dataclass
class RLTrainConfig:
    pass

class FlowTrainer:
    def __init__(self, flow: Flow, config: FlowTrainConfig):
        self.flow = flow
        self.config = config

        self.optim = self.config.get_optimizer(self.flow.parameters())
        self.dataloader = self.config.get_teacher_dataloader()
    
    def single_train_step(self, obs, action):
        """Single train step.
        
        Args:
            obs (B, obs_dim): batched observation.
            action (B, act_dim): batched action.
        
        Return:
            loss (B, 1): the conditional flow matching loss.
        """
        B, _ = obs.shape

        t = self.flow.t_sampler.sample((B, 1))
        noise = self.flow.noise_sampler.sample(action.shape)
        
        cfm_loss = self.flow.compute_cfm_loss(obs, action, noise, t, record_grad=True)
        return cfm_loss

    def train(self, epoches):
        """Imitation training.

        Args:
            epoches (int): number of epoches to train
        """
        pbar = tqdm(range(epoches))
        for epoch in pbar:
            for batch in self.dataloader:
                obs = batch['observation']
                action = batch['action']
                loss = self.single_train_step(obs, action)

                self.optim.zero_grad()
                loss.mean().backward()
                self.optim.step()

                pbar.set_description(f"Epoch: {epoch}/{epoches}, Loss: {loss.mean().item():.6f}.")
    
        print("Finish training.")
        self.save()

    def get_file_path(self):
        return os.path.join(self.config.save_dir, 'flow-imitation.pth')

    def save(self):
        os.makedirs(self.config.save_dir, exist_ok=True)
        save_fp = self.get_file_path()
        torch.save(self.flow.state_dict(), save_fp)
        print("Saved.")

    def load(self):
        self.flow.load_state_dict(torch.load(self.get_file_path()))



if __name__ == '__main__':
    pass
