'''

    AUTHOR       : Kang Mingyu (Github : ahagyue)
    DATE         : 2022.09.23
    AFFILIATION  : Seoul National University
    AIM          : DQN agent class for Atari game
    REFERENCE    : Van Hasselt, H., Guez, A., & Silver, D. (2016, March). Deep reinforcement learning with double q-learning. In Proceedings of the AAAI conference on artificial intelligence (Vol. 30, No. 1).
    
'''

import gym
import torch

from agent.dqn_agent import DqnAgent
from utils.replay.replayer_interface import ReplayInterface
from function_approximator.q_interface import Qvalue

from typing import Callable, Dict

class DDqnAgent(DqnAgent):
    def __init__(self, 
        env: gym.Env, q_val: Qvalue,
        replay_buffer: ReplayInterface, epsilon: Callable[[int], float],
        args: Dict):
        
        super().__init__(env, q_val, replay_buffer, epsilon, args)

    def behavior_policy(self, state, eps: float) -> int:
        state = torch.from_numpy(state).unsqueeze(0).type(torch.FloatTensor).to(self.args["device"])
        return self.target_q_val.action(state, eps)