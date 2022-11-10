import torch.nn as nn
from abc import ABC, abstractmethod

class Qvalue(nn.Module, ABC):
    def __init__(self):
        super(Qvalue, self).__init__()

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def action(self, x, epsilon):
        pass