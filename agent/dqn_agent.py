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
from function_approximator.q_interface import Qvalue

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
                        
                        frame_num
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

    
    # copy parameter of curr_q_val to target_q_val
    def copy_model_parameter(self):
        self.target_q_val.load_state_dict(self.curr_q_val.state_dict())

    # return action computed by curr_q_val
    def behavior_policy(self, state, eps: float) -> int:
        state = torch.from_numpy(state).unsqueeze(0).type(torch.FloatTensor).to(self.args["device"])
        return self.curr_q_val.action(state, eps)
    
    def get_replay(self, batch: int) -> Tuple:
        replay = self.replay_buffer.batch_replay(batch)
        prev_obs = torch.stack([torch.tensor(obs) for obs, _, _, _, _ in replay])
        action = torch.stack([torch.tensor([act]) for _, act, _, _, _ in replay])
        reward = torch.stack([torch.tensor([rew]) for _, _, rew, _, _ in replay])
        curr_obs = torch.stack([torch.tensor(obs) for _, _, _, obs, _ in replay])
        done = torch.stack([torch.tensor(don) for _, _, _, _, don in replay])
        
        return (
                    prev_obs.type(torch.FloatTensor).to(self.args['device']), action.to(self.args['device']),
                    reward.to(self.args['device']), curr_obs.type(torch.FloatTensor).to(self.args['device']),
                    done.type(torch.FloatTensor).to(self.args['device'])
                )
    
    def compute_loss(self):
        # get data from replay buffer
        prev_obs, actions, rewards, curr_obs, dones = self.get_replay(self.args["batch_size"])

        # loss function
        gamma = self.args["discount_factor"]
        target_q =  self.target_q_val(curr_obs).max(1)[0].unsqueeze(-1)

        expected_gain = rewards + gamma * target_q * (1 - dones)
        current_gain = self.curr_q_val(prev_obs).gather(1, actions)

        loss = (expected_gain.detach() - current_gain).pow(2).mean()

        return loss

    def training(self, verbose: bool = True):

        loss_list = []
        reward_sum_list = []
        reward_sum = 0

        observation = self.env.reset()
        for i in range(self.args["frame_num"]):
            if i % self.args["update_duration"] == 0:
                self.copy_model_parameter()

            buffer = [observation]
            # epsilon-greedy behaviour policy
            action = self.behavior_policy(observation, self.epsilon(i))
            observation, reward, done, _ = self.env.step(action)
            reward_sum += reward
            
            # save to replay buffer
            buffer += [action, reward, observation, done]
            self.replay_buffer.push(tuple(buffer))
            
            # reset environment
            if done:
                reward_sum_list.append(reward_sum)
                reward_sum = 0
                observation = self.env.reset()
            
            # train
            if len(self.replay_buffer) > self.args["replay_initial"]:
                loss = self.compute_loss()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loss_list.append(loss.item())

            if  verbose and (i+1) % 10000 == 0:
                clear_output(wait=True)
                learning_curve = Plots(fig=plt.figure(figsize=(12, 6)), subplot_num=2, position=(1, 2), suptitle="Learning Curve")
                plot(learning_curve, reward_sum_list, loss_list)
                
                torch.save({
                    'iteration': i,
                    'model_state_dict': self.curr_q_val.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': loss_list,
                    'reward': reward_sum_list
                    }, self.save_path)
    
    def get_action(self, obs):
        return self.curr_q_val.action(obs, 0)

