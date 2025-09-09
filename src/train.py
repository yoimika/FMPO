import torch
from torch import Tensor
import numpy as np
from flow import Flow, FlowConfig, Transition, make_layers
from utils import *
from tqdm import tqdm
import os
from env import *
from config.configs import FlowTrainConfig, RLTrainConfig
import argparse
from rollout import *
from losses import *
from evals import show_distribution


parser = argparse.ArgumentParser()
parser.add_argument('--il', action='store_true', help='imitation learning stage')
parser.add_argument('--train', action='store_true', help='training stage')
parser.add_argument('--vis', action='store_true', help='Use human render mode, else store mp4.')
parser.add_argument('--inst_name', default='rl1', help='instance name')

class FlowTrainer:
    def __init__(self, flow: Flow, config: FlowTrainConfig, env_config: EnvConfig):
        self.flow = flow
        self.config = config

        self.optim = self.config.get_optimizer(self.flow.parameters())
        self.dataloader = self.config.get_teacher_dataloader(env_config)
    
    def single_train_step(self, obs, action):
        """Single train step.
        
        Args:
            obs (B, obs_dim): batched observation.
            action (B, act_dim): batched action.
        
        Return:
            loss (B, 1): the conditional flow matching loss.
        """
        B, _ = obs.shape

        t = self.flow.t_sampler.sample((B, 1)).to(self.flow.device)
        noise = self.flow.noise_sampler.sample(action.shape).to(self.flow.device)
        
        cfm_loss = compute_cfm_loss(self.flow, obs, action, noise, t, record_grad=True)
        return cfm_loss

    def train(self, epoches):
        """Imitation training.

        Args:
            epoches (int): number of epoches to train
        """
        pbar = tqdm(range(epoches))
        for epoch in pbar:
            for batch in self.dataloader:
                obs = batch['observation'].to(self.flow.device)
                action = batch['action'].to(self.flow.device)
                loss = self.single_train_step(obs, action)

                self.optim.zero_grad()
                loss.mean().backward()
                self.optim.step()

                pbar.set_description(f"Epoch: {epoch}/{epoches}, Loss: {loss.mean().item():.6f}.")
    
        print("Finish training.")
        self.save()

    def get_file_path(self):
        return os.path.join(self.config.save_dir, self.config.env_name + '-' + 'il.pth')

    def save(self):
        os.makedirs(self.config.save_dir, exist_ok=True)
        save_fp = self.get_file_path()
        torch.save(self.flow.state_dict(), save_fp)
        print("Saved.")

    def load(self):
        self.flow.load_state_dict(torch.load(self.get_file_path(), map_location=device))



class RLTrainer:
    def __init__(self, flow: Flow, config: RLTrainConfig, env_config: EnvConfig):
        self.flow = flow
        self.config = config
        self.env_config = env_config
        if self.config.use_critic and not hasattr(self.flow, 'critic'):
            self.build_critic()

        self.optim = self.config.get_optimizer(self.flow.parameters())
        self.load_base_model()
    
    def build_critic(self):
        obs_dim = self.flow.config.input_dim - self.flow.config.output_dim - self.flow.config.time_embed_dim
        self.critic = make_layers([obs_dim, 256, 256, 64, 32, 1]).to(self.flow.device)
        self.critic_optim = torch.optim.AdamW(self.critic.parameters(), lr=5e-4)

    def load_base_model(self):
        if self.config.base_model_file_name:
            if len(self.config.save_dir.split('/')) == 3:
                save_dir = os.path.dirname(self.config.save_dir)
            else:
                save_dir = self.config.save_dir
            base_model_name = self.config.env_name + '-il.pth'
            fp = os.path.join(save_dir, base_model_name)
            self.flow.load_state_dict(torch.load(fp, map_location=device))
        print_green(f"Base Model Loaded Successfully. {fp}")
    
    def get_file_path(self, idx: int = None):
        save_name = self.config.env_name
        if idx is not None:
            save_fp = os.path.join(self.config.save_dir, f"{save_name}-{idx}.pth")
            return save_fp
        return os.path.join(self.config.save_dir, save_name + '-' + 'rl.pth')

    def save(self, idx:int = None):
        os.makedirs(self.config.save_dir, exist_ok=True)
        save_fp = self.get_file_path(idx)
        torch.save(self.flow.state_dict(), save_fp)
        print(f"Saved. {save_fp}")
    
    def load(self, idx: int = None):
        loaded_fp = self.get_file_path(idx)
        print(f"Load model fp: {loaded_fp}")
        state_dict = torch.load(loaded_fp, map_location=device)
        self.flow.load_state_dict(state_dict)

    def evaluate_return(self, sample_nums: int = 20):
        test_env = create_env(env_config, device=device)
        ret_mean, ret_std = compute_return(self.flow, test_env, device=device, sample_nums=sample_nums, gamma=self.env_config.env_gamma)
        test_env.close()
        return ret_mean, ret_std
    
    # PPO Train
    def train(self):
        init_mean, init_std = self.evaluate_return(sample_nums=20)
        print("Initial Average Return: %.2f ± %.2f"%(init_mean, init_std))

        env_config = replace(self.env_config, num_envs=self.config.num_envs)
        env = create_env(env_config, device=device)

        assert self.config.episode_length % (env.spec.max_episode_steps * self.config.num_envs) == 0
        iter_num = self.config.episode_length // env.spec.max_episode_steps // self.config.num_envs
        pbar = tqdm( range(self.config.epoches) )

        if self.config.use_critic:
            # 预训练 critic net，因为 flow 模型是从模仿学习过来的，初始的动作质量还不错
            pbar.set_description("Pretrain Critic...")
            pretrain_epoches = 300
            for i in range(pretrain_epoches):
                rollout_state = rollout(self.flow, env, iter_num, None, self.env_config)
                batches = rollout_state.prepare_batches(self.config.batch_size)
                critic_losses = []
                for batch in batches:
                    critic_loss = compute_critic_loss(self.critic, batch)
                    self.critic_optim.zero_grad()
                    critic_loss.backward()
                    self.critic_optim.step()
                    critic_losses.append(critic_loss.cpu().item())
                pbar.set_description(f"Pretrain Critic Epoch {i+1}/{pretrain_epoches}, Loss: {np.mean(critic_losses):.6f}")
            print(f"Finish pretraining critic.: {np.mean(critic_losses):.6f}")

        for i in pbar:
            pbar.set_description("Sampling...")
            # (T, num_envs * iter_num, dim)
            rollout_state = rollout(self.flow, env, iter_num, pbar, self.env_config)
            # (batch_size, dim) * N
            batches = rollout_state.prepare_batches(self.config.batch_size)
            pbar.set_description("Training...")

            losses = []
            for batch in batches:
                if self.config.use_critic:
                    loss, critic_loss = compute_fpo_loss(self.flow, batch, self.config, self.env_config, self.critic)
                else:
                    loss = compute_fpo_loss(self.flow, batch, self.config, self.env_config)

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                if self.config.use_critic:
                    self.critic_optim.zero_grad()
                    critic_loss.backward()
                    self.critic_optim.step()

                losses.append(loss.cpu().item())
            
            if i % 10 == 0:
                pbar.set_description("Evaluating...")
                ret_mean, ret_std = self.evaluate_return(sample_nums=10)
                print("Average Return: %.2f ± %.2f"%(ret_mean, ret_std))


            if i % self.config.save_interval == 0:
                if self.config.save_idx:
                    self.save(i)
                else:
                    self.save()
        
        self.save()


#######################################
# Training Instance
#######################################


TRAIN_MAPPING = {
    'il1': ['flow', 'il_flow_train'],
    'il2': ['flow2', 'il_flow_train2'],
    'il3': ['flow3', 'il_flow_train3'],
    'il4': ['flow4', 'il_flow_train4'],
    'il-pendulum': ['flow3', 'il_flow_train_pendulum'],
    'il-reacher': ['flow3', 'il_flow_train_reacher'],
    'il-pusher': ['flow3', 'il_flow_train_pusher'],

    'rl1': ['flow5', 'rl_flow_train'],
    'rl-sde': ['flow_sde', 'rl_flow_train_sde'],
    'rl-gpu': ['flow5', 'rl_flow_train_gpu'],
    'rl-tmp': ['flow_ode', 'rl_flow_train_tmp'],
    'rl-pendulum': ['flow3', 'rl_flow_train_pendulum'],
}

def il_train(flow_trainer: FlowTrainer):
    flow_trainer.train(epoches=flow_trainer.config.epoches)

def eval(env_config: EnvConfig, flow_trainer: FlowTrainer, eval_num: int, render: bool = False):
    eval_env(env_config, flow_trainer.flow, sample_nums=eval_num, device=device, render=render)

def rl_train(trainer: RLTrainer):
    # eval_robotics_env(trainer.config.env_name, trainer.flow, sample_nums=32, device=device, render=False)
    trainer.train()

if __name__ == '__main__':
    args = parser.parse_args()
    # Set Config
    train_config_fp = './src/config/train.yaml'
    model_config_fp = './src/config/model.yaml'
    env_config_fp = './src/config/env.yaml'

    instance_name = args.inst_name
    il_stage = args.il
    train_mode = args.train

    train_yaml_data = load_yaml(train_config_fp)
    model_yaml_data = load_yaml(model_config_fp)
    env_yaml_data = load_yaml(env_config_fp)
    eval_num = 64

    # Instantiate Env
    if train_mode:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    print(device)

    # --------------------------------------
    flow_key, flow_train_key = TRAIN_MAPPING[instance_name]
    flow_train_config = FlowTrainConfig() if il_stage else RLTrainConfig()
    overwrite_object(flow_train_config, train_yaml_data[flow_train_key])
    env_config = EnvConfig()
    overwrite_object(env_config, env_yaml_data[flow_train_config.env_name])
    env_config.update()
    env = create_env(env_config, device=device)
    flow_config = FlowConfig.build_from_env(env)
    overwrite_object(flow_config, model_yaml_data[flow_key])

    print_green(f"Env Config: ")
    print(env_config)
    print_green(f"Flow Config: ")
    print(flow_config)
    print_green(f"Flow train Config: ")
    print(flow_train_config)


    # Instantiate model
    flow = Flow(flow_config, device)
    if il_stage:
        flow_trainer = FlowTrainer(flow, flow_train_config, env_config)
        if not train_mode: # Eval
            flow_trainer.load()
    else:
        flow_trainer = RLTrainer(flow, flow_train_config, env_config)
        if not train_mode: # Eval
            flow_trainer.load(flow_train_config.load_idx)

    # Train or Eval
    if il_stage:
        if train_mode:
            il_train(flow_trainer)
        else: # Eval
            show_distribution(flow_trainer.flow, env_config)
            # eval(env_config, flow_trainer, eval_num=eval_num, render=args.vis)
    else:
        if train_mode:
            rl_train(flow_trainer)
        else: # Eval
            flow_trainer.flow.config.use_ode = True
            eval(env_config, flow_trainer, eval_num=eval_num, render=args.vis)
    
    env.close()
    
