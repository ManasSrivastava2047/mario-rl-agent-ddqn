"""
network.py

MarioNet: the CNN that approximates the action-value function Q(s, a).

DDQN uses two copies of this network:
    - online: trained via backprop every `learn_every` steps
    - target: frozen, periodically synced from `online` (see mario.py)

Input:  [batch, 4, 84, 84]  (4 stacked grayscale 84x84 frames, from wrappers.py)
Output: [batch, n_actions]  (one Q-value per action)
"""

from torch import nn


class MarioNet(nn.Module):
    """Mini CNN: (conv2d + relu) x 3 -> flatten -> (dense + relu) x 2 -> output."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, h, w = input_dim

        if h != 84:
            raise ValueError(f"Expecting input height: 84, got: {h}")
        if w != 84:
            raise ValueError(f"Expecting input width: 84, got: {w}")

        self.online = self.__build_cnn(c, output_dim)

        self.target = self.__build_cnn(c, output_dim)
        self.target.load_state_dict(self.online.state_dict())

        # Q_target parameters are frozen -- only updated by periodic sync,
        # never by backprop.
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, input, model):
        if model == "online":
            return self.online(input)
        elif model == "target":
            return self.target(input)
        else:
            raise ValueError(f"Unknown model: {model!r} (expected 'online' or 'target')")

    def __build_cnn(self, c, output_dim):
        return nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )