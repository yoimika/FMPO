import gymnasium as gym
import numpy as np


class ClippedObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(11,), dtype=np.float32)
    
    def observation(self, observation):
        obs = observation[:11]
        return obs