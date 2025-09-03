from flow import Flow, Transition, RolloutState
from dataclasses import dataclass
from config.configs import RLTrainConfig
from env import *
from tqdm import tqdm

def compute_return(trajectory: list[Transition], gamma=0.95):
    rew = 0
    for i in reversed(range(len(trajectory))):
        rew = trajectory[i].reward + gamma * rew
        trajectory[i].reward = rew

def rollout(flow: Flow, env: gym.Env, iter: int, pbar: tqdm = None):
    trajectories = []
    for i in range(iter):
        trajectory = []
        obs, _ = env.reset()
        done = np.array([0])
        while not done.sum():
            action, action_info = flow.sample_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated + truncated
            # Trick: Reward + 1 for failure 0, success 1
            trajectory.append(Transition(obs, next_obs, action, reward+1, done, action_info))
            obs = next_obs

        # import pdb;   pdb.set_trace()
        if pbar:
            pbar.set_description(f"Rollout {i+1}/{iter}")
        compute_reward_to_go = False
        if compute_reward_to_go:
            compute_return(trajectory)
        trajectories.extend(trajectory)
    return RolloutState(trajectories)

def PPO(flow: Flow, env: gym.Env, config: RLTrainConfig, optim: torch.optim.Optimizer, trainer):
    assert config.episode_length % (env.spec.max_episode_steps * config.num_envs) == 0
    iter_num = config.episode_length // env.spec.max_episode_steps // config.num_envs
    pbar = tqdm( range(config.epoches) )
    for i in pbar:
        pbar.set_description("Sampling...")
        rollout_state = rollout(flow, env, iter_num, pbar)
        # import pdb; pdb.set_trace()
        batches = rollout_state.prepare_batches(config.batch_size)
        pbar.set_description("Training...")

        losses = []
        for batch in batches:
            loss = flow.compute_fpo_loss(batch, config)

            optim.zero_grad()
            loss.backward()
            optim.step()

            losses.append(loss.cpu().item())
        
        average_reward = rollout_state.reward.mean().item()
        pbar.set_description(f"Average Reward: {average_reward:.4f}")
        print(f"Average Reward: {average_reward:.4f}, Average Loss: {losses}", flush=True)

        if i % config.save_interval == 0:
            if config.save_idx:
                trainer.save(i)
            else:
                trainer.save()