import torch.nn as nn
from abc import ABC, abstractmethod
import random

class Qvalue(nn.Module, ABC):
    def __init__(self):
        super(Qvalue, self).__init__()

    @abstractmethod
    def forward(self, x):
        pass

    def action(self, x, epsilon) -> int:
        if random.random() < epsilon:
            act = random.randint(0, self.action_number - 1)
        else:
            act = self.forward(x).argmax().item()
        return act