"""
wrappers.py

Preprocessing wrappers applied on top of the raw gym_super_mario_bros
environment before it's fed to the RL agent (mario.py / network.py).

Pipeline (matches the PyTorch Mario RL tutorial):
    raw frame [240, 256, 3] RGB
      -> SkipFrame            (repeat action `skip` times, sum reward)
      -> GrayScaleObservation  [240, 256]           (drop color channels)
      -> ResizeObservation     [84, 84]              (downsample)
      -> FrameStack            [4, 84, 84]           (stack last 4 frames)

Import make_env() from main.py to get a ready-to-train environment.
"""

import gym
import numpy as np
import torch
from gym.spaces import Box
from gym.wrappers import FrameStack
from torchvision import transforms as T

import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace


class SkipFrame(gym.Wrapper):
    """Repeat an action for `skip` frames and sum the reward.

    Consecutive frames don't differ much, so skipping saves compute
    without losing much information about the game state.
    """

    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        trunk = False
        info = {}
        obs = None
        for _ in range(self._skip):
            step_result = self.env.step(action)
            if len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
                done = bool(terminated) or bool(truncated)
                trunk = bool(truncated)
            else:
                obs, reward, done, info = step_result
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, trunk, info


class GrayScaleObservation(gym.ObservationWrapper):
    """Convert the RGB [H, W, 3] observation to grayscale [H, W]."""

    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def permute_orientation(self, observation):
        # [H, W, C] -> [C, H, W] tensor
        observation = np.transpose(observation, (2, 0, 1))
        observation = torch.tensor(observation.copy(), dtype=torch.float)
        return observation

    def observation(self, observation):
        observation = self.permute_orientation(observation)
        transform = T.Grayscale()
        observation = transform(observation)
        return observation


class ResizeObservation(gym.ObservationWrapper):
    """Resize the (grayscale) observation down to shape x shape (e.g. 84x84)."""

    def __init__(self, env, shape):
        super().__init__(env)
        self.shape = (shape, shape) if isinstance(shape, int) else tuple(shape)
        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        transforms = T.Compose(
            [T.Resize(self.shape, antialias=True), T.Normalize(0, 255)]
        )
        observation = transforms(observation).squeeze(0)
        return observation


# ---------------------------------------------------------------------------
# Action set the AGENT is allowed to choose from.
# Kept small (2 actions) on purpose -- matches the tutorial and keeps
# training tractable. Expand later (e.g. SIMPLE_MOVEMENT) once this works.
# ---------------------------------------------------------------------------
AGENT_ACTIONS = [["right"], ["right", "A"]]


def make_env(skip=4, shape=84, num_stack=4):
    """Build the fully-wrapped Super Mario Bros environment for training."""
    if gym.__version__ < "0.26":
        env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0", new_step_api=True)
    else:
        env = gym_super_mario_bros.make(
            "SuperMarioBros-1-1-v0", render_mode="rgb_array", apply_api_compatibility=True
        )

    env = JoypadSpace(env, AGENT_ACTIONS)
    env = SkipFrame(env, skip=skip)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=shape)

    if gym.__version__ < "0.26":
        env = FrameStack(env, num_stack=num_stack, new_step_api=True)
    else:
        env = FrameStack(env, num_stack=num_stack)

    return env