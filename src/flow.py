import torch 
from torch import Tensor, tensor
import torch.nn as nn
from dataclasses import dataclass, field
import gymnasium as gym

@dataclass
class ActionInfo:
    x1: Tensor # (B, sample_dim, act_dim)
    t: Tensor  # (B, sample_dim, 1)
    cfm_loss: Tensor # (B, sample_dim, 1)

@dataclass
class Transition:
    obs: torch.Tensor
    next_obs: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    done: torch.Tensor
    action_info: ActionInfo

def make_layers(layers_info: list[int]):
    layers = []
    for idx in range(len(layers_info)-1):
        in_dim, out_dim = layers_info[idx], layers_info[idx+1]

        linear_module = nn.Linear(in_dim, out_dim)
        layers.append(linear_module)
        if idx + 1 < len(layers_info) - 1:
            layers.append(nn.SELU())
    return nn.Sequential(*layers)

class TimeScheduler:
    """Time Scheduler for returning desired format time.
    """
    def __init__(self, time_steps: int, device: torch.device = torch.device('cpu')):
        self.time_steps = time_steps
        self.time_points = torch.linspace(1, 0, time_steps + 1).to(device) # Flow model t=1 for noise, t=0 for data
        self.device = device
    @property
    def curr_t(self):
        return self.time_points[:-1]
    @property
    def next_t(self):
        return self.time_points[1:]

    def get_t(self, batch_size):
        """Get batched t

        Args:
            batch_size (int)
        
        Returns:
            (steps, B, 1): batched_curr_t
            (steps, B, 1): batched_next_t
        """
        curr_t = self.curr_t.unsqueeze(-1).unsqueeze(-1).repeat(1, batch_size, 1).to(self.device)
        next_t = self.next_t.unsqueeze(-1).unsqueeze(-1).repeat(1, batch_size, 1).to(self.device)
        return curr_t, next_t


@dataclass
class FlowConfig:
    hidden_layers: list[int] = field(default_factory=lambda: [256, 256])
    input_dim: int = 789
    output_dim: int = 789

    time_embed_dim: int = 4

    time_steps: int = 10

    sde_sigma: float = 0.5 # SDE sigma term, constant.

    use_batch_norm_obs: bool = False # Batch normalize observation
    use_layer_norm_obs: bool = False # Layer normalize observation
    use_ode: bool = True # True for ODE, False for SDE
    use_mid_euler: bool = False # mid euler for v_t ((x_t' + x_t)/2) * dt else for forward euler
    use_noise_to_supervise: bool = False # use noise to supervise the flow matching loss
    use_new_t_for_training: bool = False # If true, use new time for training else sample from pervious time steps

    n_sample_dim: int = 8 # Sample times for expectation estimation.

    @staticmethod
    def build_from_env(env: gym.Env):
        ret_flow_config = FlowConfig()

        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        ret_flow_config.input_dim = obs_dim + action_dim + ret_flow_config.time_embed_dim
        ret_flow_config.output_dim = action_dim
        return ret_flow_config

    @property
    def main_layers_info(self):
        return [self.input_dim] + self.hidden_layers + [self.output_dim]
    @property
    def time_embed_layers_info(self):
        return [1] + [self.time_embed_dim]
    

class Flow(nn.Module):
    def __init__(self, config: FlowConfig, device: torch.device = torch.device('cpu')):
        super(Flow, self).__init__()
        self.config = config

        self.main_net = make_layers(config.main_layers_info)
        self.time_net = make_layers(config.time_embed_layers_info)

        self.time_scheduler: TimeScheduler = TimeScheduler(self.config.time_steps, device)
        self.t_sampler = torch.distributions.Normal(0, 1)
        self.noise_sampler = torch.distributions.Normal(0, 1)
        self.brownian_sampler = torch.distributions.Normal(0, 1)

    def forward(self, obs: Tensor, xt: Tensor, t: Tensor):
        """Forward net.
        """
        t_embed = self.time_net(t)
        x = torch.cat([obs, xt, t_embed], dim=-1)
        pred_v = self.main_net(x)
        return pred_v
    
    def euler_step(self, obs: Tensor, curr_t: Tensor, next_t: Tensor, noise: Tensor):
        """Euler steps for flow matching

        Args:
            obs (B, obs_dim)
            curr_t (B, 1)
            next_t (B, 1)
            noise (B, act_dim)

        Returns:
            (B, act_dim): Next state
            (B, 1): Used t
            (B, 1): dt 
        """
        dt = next_t - curr_t
        t = curr_t + 0.5 * dt if self.config.use_mid_euler else curr_t
        with torch.no_grad():
            pred_vel = self.forward(obs, noise, t)
        # if self.config.use_ode or (t > 0.2).all().item():
        if self.config.use_ode:
            # ODE
            dnoise = noise + dt * pred_vel
        else:
            # SDE
            sde_sigma = self.config.sde_sigma * torch.sqrt(t / (1 - t))
            brownian_eps = self.brownian_sampler.sample(noise.shape).to(noise.device)

            time_term = pred_vel + sde_sigma**2 / (2*t) * (noise + (1 - t) * pred_vel)
            # import pdb; pdb.set_trace()
            eps_term = sde_sigma * torch.sqrt(torch.abs(dt))

            dnoise = noise + time_term * dt + eps_term * brownian_eps
            # import pdb; pdb.set_trace()
        return dnoise, t, dt
    
    def compute_cfm_loss(self, obs, x0, noise, t, record_grad: bool=False):
        """Compute conditional flow matching loss

        Args:
            obs (*, ob_dim)
            x0 (*, act_dim)
            noise (*, act_dim)
            t (*, 1)
            record_grad (bool)
        
        Returns:
            (*, 1): Conditional flow matching loss
        """
        if len(obs.size()) != len(noise.size()):
            # obs, x0 (B, dim)
            # noise, t (B, T, dim)
            obs = obs.unsqueeze(1).repeat(1, noise.shape[1], 1)
            x0 = x0.unsqueeze(1).repeat(1, noise.shape[1], 1)
        xt = (1 - t) * x0 + t * noise
        pred_v = self.forward(obs, xt, t)

        if self.config.use_noise_to_supervise:
            pred_noise = xt + (1 - t) * pred_v
            cfm_loss = torch.mean((pred_noise - noise) ** 2, dim=-1, keepdim=True)  # (B, 1)
        else:
            cfm_loss = torch.mean((pred_v - noise) ** 2, dim=-1, keepdim=True)  # (B, 1)

        return cfm_loss if record_grad else cfm_loss.detach()

    def compute_adv_value(self, transition: Transition):
        """(B, 1)"""
        return transition.reward

    
    def compute_fpo_loss(self, transition: Transition, train_config):
        """Compute fpo loss.

        Args:
            transition (Transition)

        Returns: 
            torch.Tensor: FPO loss
        """
        adv = self.compute_adv_value(transition)
        if True:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        cfm_loss = self.compute_cfm_loss(transition.obs, transition.action, transition.action_info.x1, transition.action_info.t, record_grad=True).squeeze(dim=-1)
        old_cfm_loss = transition.action_info.cfm_loss.squeeze(dim=-1)
        if train_config.average_loss_before_exp:
            rho = torch.exp(
                torch.mean(old_cfm_loss, dim=-1, keepdim=True) - torch.mean(cfm_loss, dim=-1, keepdim=True)
            )
        else:
            rho = torch.exp(
                torch.clip(old_cfm_loss - cfm_loss, -3.0, 3.0)
            )
        
        surr_loss1 = rho * adv
        surr_loss2 = torch.clip(rho, 1.0 - train_config.clip_epsilon, 1.0 + train_config.clip_epsilon) * adv

        loss = -torch.mean(torch.minimum(surr_loss1, surr_loss2))
        return loss


    def sample_action(self, obs: Tensor):
        """Sample an action given the observation.

        Args:
            obs: (B, obs_dim)
        
        Returns:
            (B, act_dim): Sampled actions
            ActionInfo: Contians x1, t that can be reconstructed when training.
        """
        B = obs.shape[0]
        action_dim = self.config.output_dim

        noise = self.noise_sampler.sample((B, action_dim)).to(obs.device)
        curr_t, next_t = self.time_scheduler.get_t(B)

        x, t_path = noise, []
        for _curr_t, _next_t in zip(curr_t, next_t):
            x, t, dt = self.euler_step(obs, _curr_t, _next_t, x)
            t_path.append(t)
        t_path = torch.stack(t_path, dim=0).permute((1, 0, 2)) # (B, steps, 1)

        x1 = self.noise_sampler.sample((B, self.config.n_sample_dim, action_dim)).to(obs.device) # (B, sample_dim, act_dim)
        if self.config.use_new_t_for_training:
            t = self.t_sampler.sample((B, self.config.n_sample_dim, 1)).to(obs.device)
        else:
            t = t_path[:, torch.randint(0, t_path.shape[1], (self.config.n_sample_dim, )), :].squeeze(1) # (B, sample_dim, 1)

        loss_x = x.unsqueeze(1).repeat(1, self.config.n_sample_dim, 1)
        loss_obs = obs.unsqueeze(1).repeat(1, self.config.n_sample_dim, 1)
        ref_cfm_loss = self.compute_cfm_loss(loss_obs, loss_x, x1, t, record_grad=False) # (B, sample_dim, 1)

        return x, ActionInfo(x1=x1, t=t, cfm_loss=ref_cfm_loss)
    

    

