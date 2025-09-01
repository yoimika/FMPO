import torch
import numpy as np
from torch.utils.data import DataLoader
from flow import Flow, FlowConfig
from dataclasses import dataclass, field
from utils import *
from tqdm import tqdm
import os
from env import *
from functools import reduce

@dataclass
class FlowTrainConfig:
    learning_rate: float = 1e-4
    batch_size: int = 256
    epoches: int = 256

    optim_name: str = "Adam"

    teacher_dataset_fp: str = './save/sac-teacher.pkl' # Teacher dataset path for imitation learning.

    save_dir: str = './save/'
    save_file_name: str = 'flow-imitation.pth'

    def get_optimizer(self, params):
        if self.optim_name == "Adam":
            return torch.optim.Adam(params, lr=self.learning_rate)
        elif self.optim_name == "SGD":
            return torch.optim.SGD(params, lr=self.learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {self.optim_name}")
    
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
        return os.path.join(self.config.save_dir, self.config.save_file_name)

    def save(self):
        os.makedirs(self.config.save_dir, exist_ok=True)
        save_fp = self.get_file_path()
        torch.save(self.flow.state_dict(), save_fp)
        print("Saved.")

    def load(self):
        self.flow.load_state_dict(torch.load(self.get_file_path()))

#######################################
# Training Instance
#######################################

TRAIN_MAPPING = {
    'il1': {
        "config": ['il_train1_flow', 'il_train1_flow_train'],
        "function": ['il_train1', 'il_eval1']
    },
}

def il_train1(flow_trainer: FlowTrainer):
    flow_trainer.train(epoches=flow_trainer.config.epoches)

def il_eval1(env_id: str, env_wrappers: list, flow_trainer: FlowTrainer, eval_num: int):
    flow_trainer.load()
    env = gym.make(env_id, render_mode='human')
    env = env_wrapper(env, env_wrappers)

    eval_num += 1
    while eval_num := eval_num - 1:
        obs, _ = env.reset()
        done = False
        while not done:
            tensor_obs = from_numpy(obs).unsqueeze(0)  # (1, obs_dim)
            action, _ = flow_trainer.flow.sample_action(tensor_obs)
            action = action.cpu().squeeze().numpy()
            obs, _, terminated, truncated, _  = env.step(action)
            done = terminated or truncated
    env.close()

if __name__ == '__main__':
    # Set Config
    config_fp = './src/config/flow.yaml'
    instance_name = 'il1'
    train_mode = True
    yaml_data = load_yaml(config_fp)
    eval_num = 32
    env_wrappers = [
        (RoboticsWrapper, {}),
    ]

    # Instantiate Env
    env = gym.make(env_name := yaml_data['env_name'])
    env = env_wrapper(env, env_wrappers)

    # --------------------------------------
    flow_key, flow_train_key = TRAIN_MAPPING[instance_name]['config']
    train_func, eval_func = TRAIN_MAPPING[instance_name]['function']
    flow_config = FlowConfig.build_from_env(env)
    overwrite_object(flow_config, yaml_data[flow_key])
    flow_train_config = FlowTrainConfig()
    overwrite_object(flow_train_config, yaml_data[flow_train_key])

    print(f"Flow Config: ")
    print(flow_config)
    print(f"Flow train Config: ")
    print(flow_train_config)


    # Instantiate model
    flow = Flow(flow_config)
    flow_trainer = FlowTrainer(flow, flow_train_config)

    # Train or Eval
    if train_mode:
        eval(train_func)(flow_trainer)
    else:
        eval(eval_func)(env.spec.id, env_wrappers, flow_trainer, eval_num=eval_num)
    
    env.close()
    
