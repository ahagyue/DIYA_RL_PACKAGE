'''

    AUTHOR       : Kang Mingyu (Github : ahagyue)
    DATE         : 2022.09.23
    AFFILIATION  : Seoul National University
    AIM          : duel Q approximator
    REFERENCE    : Wang, Z., Schaul, T., Hessel, M., Hasselt, H., Lanctot, M., & Freitas, N. (2016, June). Dueling network architectures for deep reinforcement learning. In International conference on machine learning (pp. 1995-2003). PMLR.

'''

import torch.nn as nn

class DuelNet(nn.Module):
    '''
    for dueling the network
    Q(s, a) = V(s) + A(s, a) -> V(s) + ( A(s, a) - avg_a(A(s, a)) )
    '''
    def __init__(self, input_dim: int, action_number:int=6):
        super(DuelNet, self).__init__()

        self.input_dim = input_dim
        self.value_in_dim = input_dim // 2
        self.advantage_in_dim = (input_dim + 1) // 2

        self.value_approximator = nn.Linear(self.value_in_dim, 1)
        self.advantage_approximator = nn.Linear(self.advantage_in_dim, action_number)

    def get_value(self, x):
        value = self.value_approximator(x[:, :self.value_in_dim]) 
        return value
    
    def get_advantage(self, x):
        advantage = self.advantage_approximator(x[:, -self.advantage_in_dim:])
        return advantage - advantage.mean(1, True)
    
    def forward(self, x):
        q_val = self.get_value(x) + self.get_advantage(x)
        return q_val