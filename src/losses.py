import torch
import numpy as np
import torch.nn as nn
import pickle

def compute_cfm_loss(flow, obs, x0, noise, t, record_grad: bool=False):
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
    pred_v = flow.forward(obs, xt, t)

    if flow.config.use_noise_to_supervise:
        pred_noise = xt + (1 - t) * pred_v
        cfm_loss = torch.mean((pred_noise - noise) ** 2, dim=-1, keepdim=True)  # (B, 1)
    else:
        cfm_loss = torch.mean((pred_v - noise) ** 2, dim=-1, keepdim=True)  # (B, 1)

    return cfm_loss if record_grad else cfm_loss.detach()

def compute_critic_value(critic, obs, return_grad: bool = False):
    assert critic is not None, "Critic not built."
    ret_value = critic(obs)
    if return_grad:
        return ret_value
    else:
        return ret_value.detach()
    

def compute_adv_value(critic, transition, gamma):
    """(B, 1)"""
    if critic is None:
        return transition.reward
    else:
        curr_value = compute_critic_value(critic, transition.obs)
        # return_value = transition.reward_to_go - curr_value
        # return return_value
        next_value = compute_critic_value(critic, transition.next_obs)
        adv = transition.reward.unsqueeze(-1) + gamma * next_value * (1 - transition.done.unsqueeze(-1))
        return adv - curr_value

def compute_critic_loss(critic, transition):
    assert critic is not None, "Critic not built."
    curr_value = compute_critic_value(critic, transition.obs, return_grad=True).squeeze(dim=-1)

    next_value = compute_critic_value(critic, transition.next_obs).squeeze(dim=-1)
    critic_loss = nn.MSELoss()(transition.reward + 0.95 * next_value * (1 - transition.done), curr_value)
    # critic_loss = nn.MSELoss()(curr_value, transition.reward_to_go)
    return critic_loss

def compute_fpo_loss(flow, transition, train_config, env_config, critic = None):
    """Compute fpo loss.

    Args:
        transition (Transition)

    Returns: 
        torch.Tensor: FPO loss
    """
    adv = compute_adv_value(critic, transition, env_config.env_gamma)
    if True:
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    cfm_loss = compute_cfm_loss(flow, transition.obs, transition.action, transition.action_info.x1, transition.action_info.t, record_grad=True).squeeze(dim=-1)
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

    loss = torch.mean(torch.minimum(surr_loss1, surr_loss2))

    # Compute Critic
    if critic is not None:
        critic_loss = compute_critic_loss(critic, transition)
        # import pdb; pdb.set_trace()
        return loss, critic_loss

    return loss

def compute_return(flow, env, device:torch.device = torch.device('cpu'), sample_nums: int = 20, gamma: float = 0.95):
    with torch.no_grad():
        rets = []
        while sample_nums:
            obs, _ = env.reset()
            done = torch.zeros(0, dtype=torch.float32)
            rews = []
            while not done.sum():
                if len(obs.shape) == 1:
                    obs = obs.unsqueeze(0)
                action, _ = flow.sample_action(obs.to(device))  # (1, action_dim)
                obs, rew, terminated, truncated, info = env.step(action.cpu().squeeze(0).numpy())
                done = (terminated + truncated)
                rews.append(rew.cpu().numpy() if isinstance(rew, torch.Tensor) else rew)
            
            for i in reversed(range(len(rews))):
                rews[i] = rews[i] + (rews[i+1] if i+1 < len(rews) else 0) * gamma
            rets.append(rews[0])

            sample_nums -= 1
    
    rets = torch.tensor(rets)
    return rets.mean().item(), rets.std().item()


def compute_expert_return(expert_fp: str):
    data = pickle.load(open(expert_fp, 'rb'))

    rets = []
    for traj in data:
        ret = 0
        for transition in reversed(traj):
            ret = transition['reward'] + ret * 0.95
        rets.append(ret)
    
    print("Expert Average Return: %.2f ± %.2f"%(np.mean(rets), np.std(rets)))

if __name__ == '__main__':
    fp = './save/pendulum-teacher.pkl'
    compute_expert_return(fp)
