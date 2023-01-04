import numpy as np
import matplotlib.pyplot as plt

from common.plot import Plots
from typing import List

def plot_learning_curve(sketch_book: Plots, rewards: List=None, losses: List=None, legend:list = None):
    if rewards is not None:
        rewards = np.array(rewards)
        reward_xval = np.array(
                  range(rewards.size) if rewards.ndim==1
                  else [range(rewards[i].size) for i in range(rewards.shape[0])])
                  
        sketch_book.repair_to_graph(
                                        plot_num=0,
                                        x_value=reward_xval, y_value=rewards,
                                        title="reward", x_label="episode", y_label="reward", graph_name=legend
                                    )
    if losses is not None:
        losses = np.array(losses)
        loss_xval = np.array(
                  range(losses.size) if losses.ndim==1
                  else [range(losses[i].size) for i in range(losses.shape[0])])

        sketch_book.repair_to_graph(
                                        plot_num=1,
                                        x_value=loss_xval, y_value=losses,
                                        title="loss", x_label="iteration", y_label="loss", graph_name=legend
                                    )
    plt.pause(0.1)

def smoothing(array:list, window:int =  30):
    ret = []
    for i in range(window//2, len(array) - window//2):
        ret.append(sum(array[i - window//2 : i + window//2]) / window)
    return ret