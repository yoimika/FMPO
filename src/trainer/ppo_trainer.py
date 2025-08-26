import os
from base_trainer import BaseTrainer
from dataclasses import dataclass
from agents.base_agent import BaseAgent
from agents.ppo import PPO, PPOConfig
from utils.buffer import Buffer
from envs import ENV_MAPS
import numpy as np
import torch

from utils.dtype import list2torch, torch2numpy

@dataclass
class PPOTrainerConfig:
    learning_rate: float = 1e-4
    n_epochs: int = 10000
    inner_update_steps: int = 2

    clip: float = 0.1

    env: str = "FetchPickAndPlaceDense-v4"

    episodes_per_epoch: int = 30
    gamma: float = 0.95

    ppo_config: PPOConfig = None

    save_interval: int = 100
    save_fp: str = '/home/yoimisan/projects/FMPO/save/ppo'



class PPOTrainer(BaseTrainer):
    def __init__(self, config: PPOTrainerConfig):
        super().__init__()
        self.config = config

        self._create_all()
        self.buffer = Buffer()
    
    def _create_all(self):
        self.env = ENV_MAPS[self.config.env]()
        self.agent = PPO(self.config.ppo_config)

        self.actor_optim = torch.optim.Adam(self.agent.actor.parameters(), lr=self.config.learning_rate)
        self.critic_optim = torch.optim.Adam(self.agent.critic.parameters(), lr=self.config.learning_rate)
    

    def train(self):
        ave_rews = []
        for epoch in range(self.config.n_epochs):
            # Sampling
            self.buffer.clear()
            for episode_idx in range(self.config.episodes_per_epoch):
                epi_obs, epi_act, epi_rew, epi_next_obs, epi_done = [], [], [], [], []
                epi_log_prob = []
                obs, _ = self.env.reset()
                while True:
                    action, log_prob = self.agent.sample_action(obs)
                    next_obs, reward, terminated, truncated, info = self.env.step(torch2numpy(action))
                    done = terminated or truncated

                    epi_obs.append(obs)
                    epi_rew.append(reward)
                    epi_done.append(done)
                    epi_next_obs.append(next_obs)
                    epi_act.append(action)
                    epi_log_prob.append(log_prob)

                    if done:
                        break
                    obs = next_obs

                self.buffer.push({
                    "observations": list2torch(epi_obs),
                    "actions": list2torch(epi_act),
                    "rewards": list2torch(epi_rew),
                    "next_observations": list2torch(epi_next_obs),
                    "dones": list2torch(epi_done),
                    "log_probs": list2torch(epi_log_prob),
                })
            
            self.buffer.compute_rewards_to_go(self.config.gamma)

            # Updating
            all_rewards_to_go = self.buffer.get_all_use_key("rewards_to_go")
            all_observations = self.buffer.get_all_use_key("observations")
            all_actions = self.buffer.get_all_use_key("actions")
            all_log_probs = self.buffer.get_all_use_key("log_probs")
            v = self.agent.get_value(all_observations).detach()
            adv = all_rewards_to_go - v
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            for _ in range(self.config.inner_update_steps):
                log_probs = self.agent.get_log_prob(all_observations, all_actions)
                ratios = torch.exp(log_probs - all_log_probs)

                surr1 = ratios * adv
                surr2 = torch.clamp(ratios, 1.0 - self.config.clip, 1.0 + self.config.clip) * adv

                v = self.agent.get_value(all_observations)

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = torch.nn.MSELoss()(all_rewards_to_go, v)

                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optim.step()

                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

            ave_rew = self.buffer.get_all_use_key("rewards").mean()
            print(f"Epoches: {epoch:05d}/{self.config.n_epochs:05d} | Average rewards: {ave_rew:.4f}")
            ave_rews.append(ave_rew.cpu().item())

            if epoch % self.config.save_interval == 0 or epoch + 1 == self.config.n_epochs:
                save_fp = os.path.join(self.config.save_fp, self.config.env)
                os.makedirs(save_fp, exist_ok=True)
                save_dict = {
                    "agent": self.agent.state_dict(),
                    "ave_rewards": ave_rews,
                }
                torch.save(save_dict, f"{save_fp}/ppo_{self.config.env}_{epoch}.pth")
    
    def eval(self):
        files = sorted(os.listdir(self.config.save_fp), key=lambda x : int(x.split('_')[-1][:-4]))
        trgt_file_path = os.path.join(self.config.save_fp, files[-1])
        data_dict = torch.load(trgt_file_path)

        self.agent.load_state_dict(data_dict['agent'])
        
        eval_env = ENV_MAPS[self.config.env](render_mode='human')
        for _ in range(3):
            obs, _ = eval_env.reset()
            while True:
                act, _ = self.agent.sample_action(obs)
                obs, reward, terminated, truncated, _ = eval_env.step(act)
                if terminated or truncated:
                    break
        eval_env.close()


    # def evaluate(self):
    #     pass
if __name__ == '__main__':
    env_name = "Pendulum-v1"
    obj = ENV_MAPS[env_name]()
    obs_dim = obj.observation_space.shape[0]
    act_dim = obj.action_space.shape[0]
    ppo_config = PPOConfig(
        actor_layers=[obs_dim, 64, 64, act_dim],
        critic_layers=[obs_dim, 64, 64, 1],
    )

    trainer_config = PPOTrainerConfig(
        ppo_config=ppo_config,
        env=env_name
    )
    
    trainer = PPOTrainer(trainer_config)
    trainer.train()
    # trainer.eval()

