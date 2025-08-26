import gymnasium as gym
import gymnasium_robotics

from wrapper.flatten_observation_wrapper import FlattenObservationWrapper
from wrapper.clipped_observation_wrapper import ClippedObservationWrapper
from wrapper.fetch_pick_and_place_v4_wrapper import FetchPickAndPlaceV4Wrapper



gym.register_envs(gymnasium_robotics)

def FetchPickAndPlaceV4(*args, **kwargs):
    env = gym.make("FetchPickAndPlace-v4", **kwargs)
    env = FlattenObservationWrapper(env)
    return env

def FetchPickAndPlaceDenseV4(*args, **kwargs):
    env = gym.make("FetchPickAndPlace-v4", **kwargs)
    env = FetchPickAndPlaceV4Wrapper(env)
    return env

def PendulumV1(*args, **kwargs):
    env = gym.make("Pendulum-v1", **kwargs)
    return env

ENV_MAPS = {
    "FetchPickAndPlace-v4": FetchPickAndPlaceV4,
    "FetchPickAndPlaceDense-v4": FetchPickAndPlaceDenseV4,
    "Pendulum-v1": PendulumV1,
}