import torch

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
    

def compute_adv_value(critic, transition):
    """(B, 1)"""
    if critic is None:
        return transition.reward
    else:
        curr_value = compute_critic_value(critic, transition.obs)
        next_value = compute_critic_value(critic, transition.next_obs)
        adv = transition.reward + flow.config.gamma * next_value * (1 - transition.done)
        return adv - curr_value


def compute_fpo_loss(flow, transition, train_config, critic = None):
    """Compute fpo loss.

    Args:
        transition (Transition)

    Returns: 
        torch.Tensor: FPO loss
    """
    adv = compute_adv_value(critic, transition)
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

    loss = -torch.mean(torch.minimum(surr_loss1, surr_loss2))

    # Compute Critic
    if critic is not None:
        critic_value = compute_critic_value(critic, transition.obs, return_grad=True)
        critic_loss = nn.MSELoss()(critic_value, transition.reward_to_go)
        return loss, critic_loss

    return loss
