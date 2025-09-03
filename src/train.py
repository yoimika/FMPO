import torch
import numpy as np
from flow import Flow, FlowConfig
from utils import *
from tqdm import tqdm
import os
from env import *
from config.configs import FlowTrainConfig, RLTrainConfig
import argparse
from algorithm import PPO


parser = argparse.ArgumentParser()
parser.add_argument('--il', action='store_true', help='imitation learning stage')
parser.add_argument('--train', action='store_true', help='training stage')
parser.add_argument('--eval', action='store_true', help='training stage')
parser.add_argument('--vis', action='store_true', help='training stage')
parser.add_argument('--inst_name', default='rl1', help='instance name')

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



class RLTrainer:
    def __init__(self, flow: Flow, config: RLTrainConfig):
        self.flow = flow
        self.config = config

        self.optim = self.config.get_optimizer(self.flow.parameters())
        self.load_base_model()
    
    def load_base_model(self):
        if self.config.base_model_file_name:
            fp = os.path.join(self.config.save_dir, self.config.base_model_file_name)
            self.flow.load_state_dict(torch.load(fp))
        print_green(f"Base Model Loaded Successfully. {fp}")
    
    def get_file_path(self, idx: int = None):
        if idx is not None:
            save_name, suffix = self.config.save_file_name.split('.')
            save_fp = os.path.join(self.config.save_dir, f"{save_name}_{idx}.{suffix}")
            return save_fp
        return os.path.join(self.config.save_dir, self.config.save_file_name)

    def save(self, idx:int = None):
        os.makedirs(self.config.save_dir, exist_ok=True)
        save_fp = self.get_file_path(idx)
        torch.save(self.flow.state_dict(), save_fp)
        print("Saved.")
    
    def load(self, idx: int = None):
        loaded_fp = self.get_file_path(idx)
        print(f"Load model fp: {loaded_fp}")
        state_dict = torch.load(loaded_fp, map_location=device)
        self.flow.load_state_dict(state_dict)
    
    # PPO Train
    def train(self):
        env = self.config.get_env(device)
        if self.config.algorithm == "PPO":
            PPO(self.flow, env, self.config, self.optim, self)
        
        self.save()


#######################################
# Training Instance
#######################################


TRAIN_MAPPING = {
    'il1': ['flow', 'il_flow_train'],
    'il2': ['flow2', 'il_flow_train2'],
    'il3': ['flow3', 'il_flow_train3'],
    'il4': ['flow4', 'il_flow_train4'],

    'rl1': ['flow5', 'rl_flow_train'],
    'rl_sde': ['flow_sde', 'rl_flow_train_sde'],
    'rl_gpu': ['flow_ode', 'rl_flow_train_gpu'],
    'rl_tmp': ['flow_ode', 'rl_flow_train_tmp'],
}

def il_train(flow_trainer: FlowTrainer):
    flow_trainer.train(epoches=flow_trainer.config.epoches)

def eval(env_id: str, flow_trainer: FlowTrainer, eval_num: int, render: bool = False):
    eval_robotics_env(env_id, flow_trainer.flow, sample_nums=eval_num, device=device, render=render)

def rl_train(trainer: RLTrainer):
    eval_robotics_env(trainer.config.env_name, trainer.flow, sample_nums=128, device=device, render=False)
    trainer.train()

if __name__ == '__main__':
    args = parser.parse_args()
    # Set Config
    config_fp = './src/config/flow.yaml'
    instance_name = args.inst_name
    il_stage = args.il
    train_mode = args.train
    yaml_data = load_yaml(config_fp)
    eval_num = 64

    print_green(f"YAML data:")
    print(yaml_data)

    # Instantiate Env
    if train_mode:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    print(device)
    env = create_robotics_env(yaml_data['env_name'], vec=False, device=device)

    # --------------------------------------
    flow_key, flow_train_key = TRAIN_MAPPING[instance_name]
    flow_config = FlowConfig.build_from_env(env)
    overwrite_object(flow_config, yaml_data[flow_key])
    flow_train_config = FlowTrainConfig() if il_stage else RLTrainConfig()
    overwrite_object(flow_train_config, yaml_data[flow_train_key])

    print_green(f"Flow Config: ")
    print(flow_config)
    print_green(f"Flow train Config: ")
    print(flow_train_config)


    # Instantiate model
    flow = Flow(flow_config, device).to(device)
    if il_stage:
        flow_trainer = FlowTrainer(flow, flow_train_config)
        if not train_mode:
            flow_trainer.load()
    else:
        flow_trainer = RLTrainer(flow, flow_train_config)
        if not train_mode:
            flow_trainer.load(None if flow_train_config.load_idx is None else flow_train_config.load_idx)

    # Train or Eval
    if il_stage:
        if train_mode:
            il_train(flow_trainer)
        else:
            eval(env.spec.id, flow_trainer, eval_num=eval_num, render=args.vis)
    else:
        if train_mode:
            rl_train(flow_trainer)
        else:
            flow_trainer.flow.config.use_ode = True
            eval(env.spec.id, flow_trainer, eval_num=eval_num, render=args.vis)
    
    env.close()
    
