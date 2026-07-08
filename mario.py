"""
mario.py

The Mario agent: acts in the environment, remembers experience, and learns
a Q-function via Double Deep Q-Learning (DDQN).

Usage (see main.py):
    mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir)
    action = mario.act(state)
    mario.cache(state, next_state, action, reward, done)
    q, loss = mario.learn()
"""

import numpy as np
import torch
from tensordict import TensorDict
from torchrl.data import LazyMemmapStorage, TensorDictReplayBuffer

from network import MarioNet


class Mario:
    def __init__(self, state_dim, action_dim, save_dir):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Mario's DNN to predict the most optimal action
        self.net = MarioNet(self.state_dim, self.action_dim).float()
        self.net = self.net.to(device=self.device)

        # --- Exploration (epsilon-greedy) ---
        self.exploration_rate = 1
        self.exploration_rate_decay = 0.99999975
        self.exploration_rate_min = 0.1
        self.curr_step = 0
        self.training = True

        # --- Replay memory ---
        # NOTE: LazyMemmapStorage pre-allocates its full capacity as a
        # memory-mapped file. On Windows this is backed by the system
        # paging file (unlike Linux), so a capacity of 100000 (~20+ GB for
        # [4,84,84] frames) will throw "WinError 1455: The paging file is
        # too small" unless you've manually enlarged your page file.
        # 20000 (~4-5 GB) is a safer default for a CPU-only Windows setup;
        # raise it later once training is confirmed to work end-to-end.
        self.memory = TensorDictReplayBuffer(
            storage=LazyMemmapStorage(20000, device=torch.device("cpu"))
        )
        self.batch_size = 32

        # --- DDQN learning hyperparameters ---
        self.gamma = 0.9
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025)
        self.loss_fn = torch.nn.SmoothL1Loss()

        # --- Training schedule ---
        self.burnin = 1e4       # min. experiences before training starts
        self.learn_every = 3    # no. of experiences between Q_online updates
        self.sync_every = 1e4   # no. of experiences between Q_target <- Q_online sync
        self.save_every = 5e5   # no. of experiences between checkpoints

    # ------------------------------------------------------------------
    # Act
    # ------------------------------------------------------------------
    def act(self, state):
        """Given a state, choose an epsilon-greedy action and update step count.

        Inputs:
            state (LazyFrame): a single observation of the current state
        Outputs:
            action_idx (int): index of the action Mario will perform
        """
        if np.random.rand() < self.exploration_rate:
            # EXPLORE
            action_idx = np.random.randint(self.action_dim)
        else:
            # EXPLOIT
            state = state[0].__array__() if isinstance(state, tuple) else state.__array__()
            state = torch.tensor(state, device=self.device).unsqueeze(0)
            action_values = self.net(state, model="online")
            action_idx = torch.argmax(action_values, axis=1).item()

        # decay exploration rate (only while training)
        if self.training:
            self.exploration_rate *= self.exploration_rate_decay
            self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        self.curr_step += 1
        return action_idx

    # ------------------------------------------------------------------
    # Cache and recall (replay buffer)
    # ------------------------------------------------------------------
    def cache(self, state, next_state, action, reward, done):
        """Store an experience tuple in the replay buffer."""

        def first_if_tuple(x):
            return x[0] if isinstance(x, tuple) else x

        state = first_if_tuple(state).__array__()
        next_state = first_if_tuple(next_state).__array__()

        state = torch.tensor(state)
        next_state = torch.tensor(next_state)
        action = torch.tensor([action])
        reward = torch.tensor([reward])
        done = torch.tensor([done])

        self.memory.add(
            TensorDict(
                {
                    "state": state,
                    "next_state": next_state,
                    "action": action,
                    "reward": reward,
                    "done": done,
                },
                batch_size=[],
            )
        )

    def recall(self):
        """Sample a batch of experiences from the replay buffer."""
        batch = self.memory.sample(self.batch_size).to(self.device)
        state, next_state, action, reward, done = (
            batch.get(key) for key in ("state", "next_state", "action", "reward", "done")
        )
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

    # ------------------------------------------------------------------
    # DDQN math
    # ------------------------------------------------------------------
    def td_estimate(self, state, action):
        current_Q = self.net(state, model="online")[np.arange(0, self.batch_size), action]
        return current_Q

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        next_state_Q = self.net(next_state, model="online")
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.net(next_state, model="target")[np.arange(0, self.batch_size), best_action]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()

    def update_Q_online(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------
    def save(self, filename="mario_latest.chkpt", episode=None):
        save_path = self.save_dir / filename
        torch.save(
            {
                "model": self.net.state_dict(),
                "exploration_rate": self.exploration_rate,
                "curr_step": self.curr_step,
                "episode": episode,
            },
            save_path,
        )
        print(f"MarioNet saved to {save_path} (step {self.curr_step}, episode {episode})")
        return save_path

    def load(self, path, exploit_only=False):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.net.load_state_dict(checkpoint["model"])
        self.net.target.load_state_dict(self.net.online.state_dict())
        self.exploration_rate = checkpoint.get("exploration_rate", self.exploration_rate)
        self.curr_step = checkpoint.get("curr_step", 0)
        if exploit_only:
            self.exploration_rate = 0.0
            self.training = False
        return checkpoint.get("episode", 0)

    # ------------------------------------------------------------------
    # Learn (ties everything together -- called once per environment step)
    # ------------------------------------------------------------------
    def learn(self):
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        if self.curr_step % self.save_every == 0:
            self.save()

        if self.curr_step < self.burnin:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None

        state, next_state, action, reward, done = self.recall()

        td_est = self.td_estimate(state, action)
        td_tgt = self.td_target(reward, next_state, done)
        loss = self.update_Q_online(td_est, td_tgt)

        return (td_est.mean().item(), loss)