'''

    AUTHOR       : Kang Mingyu (Github : ahagyue)
    DATE         : 2022.09.23
    AFFILIATION  : Seoul National University
    AIM          : DQN agent class for Atari game
    REFERENCE    : Mnih, V., Kavukcuoglu, K., Silver, D. et al. Human-level control through deep reinforcement learning. Nature 518, 529–533 (2015).

'''

import torch
import torch.nn as nn
import gym
import matplotlib.pyplot as plt

from common.plot import Plots
from common.common import plot_learning_curve as plot
from utils.replay.replayer_interface import ReplayInterface
from utils.replay.replay_format import get_replay
from function_approximator.q_interface import Qvalue

import os
from IPython.display import clear_output
from typing import Callable, Dict, Tuple

class DqnAgent:
    def __init__(self, 
        env: gym.Env, q_val: Qvalue,
        replay_buffer: ReplayInterface, epsilon: Callable[[int], float],
        args: Dict):
        '''
        env:            environment
        q_val:          Q value function approximator class

        replay_buffer:  replay que which implements ReplayInterface
        epsilon:        used in epsilon-greedy algorithm

        args:
                        USE_GPU
                        GPU_NUM
                        device
                        
                        episode_num
                        learning_rate
                        discount_factor
                        update_duration
        '''

        self.env = env
        self.target_q_val = q_val().eval().to(args["device"])
        self.curr_q_val = q_val().to(args["device"])

        self.replay_buffer = replay_buffer
        self.epsilon = epsilon

        self.args = args

        self.optimizer = torch.optim.Adam(self.curr_q_val.parameters(), lr = self.args["learning_rate"])
        self.save_path = args["model_path"] + args["model_name"] + ".pt"

        # for training
        self.start_episode = 0
        self.frame_num = 0
        self.loss_list=[]
        self.reward_sum_list=[]

    
    # copy parameter of curr_q_val to target_q_val
    def copy_model_parameter(self):
        self.target_q_val.load_state_dict(self.curr_q_val.state_dict())

    # return action computed by curr_q_val
    def behavior_policy(self, state, eps: float) -> int:
        state = torch.from_numpy(state).unsqueeze(0).type(torch.FloatTensor).to(self.args["device"])
        return self.curr_q_val.action(state, eps)
    
    def compute_loss(self):
        # get data from replay buffer
        prev_obs, actions, rewards, curr_obs, dones = get_replay(self.args["batch_size"], self.replay_buffer)

        # loss function
        gamma = self.args["discount_factor"]
        target_q =  self.target_q_val(curr_obs).max(1)[0].unsqueeze(-1)

        expected_gain = rewards + gamma * target_q * (1 - dones)
        current_gain = self.curr_q_val(prev_obs).gather(1, actions)

        loss = (expected_gain.detach() - current_gain).pow(2).mean()

        return loss

    def training(self, verbose: bool = True):
        self.load_model()
        reward_sum = 0
        observation = self.env.reset()
        
        for episode in range(self.start_episode, self.args["episode_num"]):
            done = False
            while not done:
                if self.frame_num % self.args["update_duration"] == 0:
                    self.copy_model_parameter()

                buffer = [observation]
                # epsilon-greedy behaviour policy
                action = self.behavior_policy(observation, self.epsilon(self.frame_num))
                observation, reward, done, _ = self.env.step(action)
                reward_sum += reward
                
                # save to replay buffer
                buffer += [action, reward, observation, done]
                self.replay_buffer.push(tuple(buffer))
                
                # train
                if len(self.replay_buffer) > self.args["replay_initial"]:
                    loss = self.compute_loss()
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    self.loss_list.append(loss.item())
                self.frame_num += 1
            
            # reset environment
            self.reward_sum_list.append(reward_sum)
            reward_sum = 0
            observation = self.env.reset()

            if  verbose and (episode) % 10 == 0:
                clear_output(wait=True)
                learning_curve = Plots(fig=plt.figure(figsize=(12, 6)), subplot_num=2, position=(1, 2), suptitle="Learning Curve")
                plot(learning_curve, self.reward_sum_list, self.loss_list)
                
                torch.save({
                    'episode': episode,
                    'frame_num': self.frame_num,
                    'current_model_state_dict': self.curr_q_val.state_dict(),
                    'target_model_state_dict': self.target_q_val.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': self.loss_list,
                    'reward': self.reward_sum_list
                    }, self.save_path)
    
    def load_model(self):
        if not os.path.isfile(self.save_path): return
        checkpoint = torch.load(self.save_path)
        self.start_episode = checkpoint["episode"] + 1
        self.frame_num = checkpoint["frame_num"]
        self.curr_q_val.load_state_dict(checkpoint["current_model_state_dict"])
        self.target_q_val.load_state_dict(checkpoint["target_model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.loss_list = checkpoint["loss"]
        self.reward_sum_list = checkpoint["reward"]

    def get_action(self, obs):
        return self.curr_q_val.action(obs, 0)

