import torch
import numpy as np
from flow import Flow, FlowConfig
from utils import *
from tqdm import tqdm
import os
from env import *
from config.configs import FlowTrainConfig, RLTrainConfig
from algorithm import PPO


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
        print_green("Base Model Loaded Successfully.")
    
    def get_file_path(self):
        return os.path.join(self.config.save_dir, self.config.save_file_name)

    def save(self):
        os.makedirs(self.config.save_dir, exist_ok=True)
        save_fp = self.get_file_path()
        torch.save(self.flow.state_dict(), save_fp)
        print("Saved.")
    
    def load(self):
        self.flow.load_state_dict(torch.load(self.get_file_path()))
    
    # PPO Train
    def train(self):
        env = self.config.get_env()
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

    'rl1': ['flow5', 'rl_flow_train']
}

def il_train(flow_trainer: FlowTrainer):
    flow_trainer.train(epoches=flow_trainer.config.epoches)

def eval(env_id: str, env_wrappers: list, flow_trainer: FlowTrainer, eval_num: int):
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

def rl_train(trainer: RLTrainer):
    trainer.train()

if __name__ == '__main__':
    # Set Config
    config_fp = './src/config/flow.yaml'
    instance_name = 'rl1'
    il_stage = False
    train_mode = False
    yaml_data = load_yaml(config_fp)
    eval_num = 32
    env_wrappers = [
        (RoboticsWrapper, {}),
    ]

    print_green(f"YAML data:")
    print(yaml_data)

    # Instantiate Env
    env = gym.make(env_name := yaml_data['env_name'])
    env = env_wrapper(env, env_wrappers)

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
    flow = Flow(flow_config)
    if il_stage:
        flow_trainer = FlowTrainer(flow, flow_train_config)
    else:
        flow_trainer = RLTrainer(flow, flow_train_config)

    # Train or Eval
    if il_stage:
        if train_mode:
            il_train(flow_trainer)
        else:
            eval(env.spec.id, env_wrappers, flow_trainer, eval_num=eval_num)
    else:
        if train_mode:
            rl_train(flow_trainer)
        else:
            flow_trainer.flow.config.use_ode = True
            eval(env.spec.id, env_wrappers, flow_trainer, eval_num=eval_num)
    
    env.close()
    
