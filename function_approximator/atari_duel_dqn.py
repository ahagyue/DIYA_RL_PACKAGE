'''

    AUTHOR       : Kang Mingyu (Github : ahagyue)
    DATE         : 2022.09.23
    AFFILIATION  : Seoul National University
    AIM          : duel deep Q network for Atari game
    REFERENCE    : Wang, Z., Schaul, T., Hessel, M., Hasselt, H., Lanctot, M., & Freitas, N. (2016, June). Dueling network architectures for deep reinforcement learning. In International conference on machine learning (pp. 1995-2003). PMLR.

'''

import torch
import torch.nn as nn

import random

from function_approximator.duel_net import DuelNet
from function_approximator.q_interface import Qvalue

class AtariDuelDQN(Qvalue):
    def __init__(self, frame_size=84, action_number=6):
        super(Qvalue, self).__init__()

        self.action_number = action_number
        
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        in_features = ((((frame_size - 8) // 4 - 3) // 2 + 1) - 2) **2 * 64
        
        self.fc_layer = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=512),
            nn.ReLU(),
        )
        self.duel_net = DuelNet(512, action_number)
    
    def forward(self, x):
        x = self.conv_layer(x)
        x = self.fc_layer(torch.flatten(x, 1))
        return self.duel_net(x)

    def action(self, x, epsilon) -> int:
        if random.random() < epsilon:
            act = random.randint(0, self.action_number - 1)
        else:
            act = self.forward(x).argmax().item()
        return act