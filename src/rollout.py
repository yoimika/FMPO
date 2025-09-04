from flow import Flow, ActionInfo, Transition
from dataclasses import dataclass, field
import torch

class RolloutState(Transition):
    def __init__(self, transitions: list[Transition]):
        def stack_tensor(attr):
            return torch.stack(attr, dim=0)
        self.obs = stack_tensor([t.obs for t in transitions])
        self.next_obs = stack_tensor([t.next_obs for t in transitions])
        self.action = stack_tensor([t.action for t in transitions])
        self.reward = stack_tensor([t.reward for t in transitions])
        self.done = stack_tensor([t.done for t in transitions])
        self.action_info = ActionInfo(
            cfm_loss=stack_tensor([t.action_info.cfm_loss for t in transitions]),
            t=stack_tensor([t.action_info.t for t in transitions]),
            x1=stack_tensor([t.action_info.x1 for t in transitions])
        )
    
    def prepare_batches(self, batch_size):
        T, B, _ = self.obs.shape
        # print(T, B)
        assert T * B % batch_size == 0
        length = T * B // batch_size

        def _prepare_single_batches(item):
            suffix = item.shape[2:]
            return item.view(length, batch_size, *suffix)

        obs = _prepare_single_batches(self.obs)
        next_obs = _prepare_single_batches(self.next_obs)
        action = _prepare_single_batches(self.action)
        reward = _prepare_single_batches(self.reward)
        done = _prepare_single_batches(self.done)

        cfm_loss = _prepare_single_batches(self.action_info.cfm_loss)
        t = _prepare_single_batches(self.action_info.t)
        x1 = _prepare_single_batches(self.action_info.x1)

        return [
            Transition(
                obs=obs[i], 
                next_obs=next_obs[i], 
                action=action[i], 
                reward=reward[i], 
                done=done[i], 
                action_info=ActionInfo(
                    cfm_loss=cfm_loss[i], 
                    t=t[i], 
                    x1=x1[i]
                )
            ) for i in range(length)
        ]