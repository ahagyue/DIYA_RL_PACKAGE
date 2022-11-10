import numpy as np
import matplotlib.pyplot as plt

from common.plot import Plots
from typing import List

def plot_learning_curve(sketch_book: Plots, rewards: List, losses: List):

    sketch_book.repair_to_graph(
                                            plot_num=0,
                                            x_value=np.array(range(len(rewards))), y_value=np.array(rewards),
                                            title="reward", x_label="episode", y_label="reward"
                                        )
    sketch_book.repair_to_graph(
                                            plot_num=1,
                                            x_value=np.array(range(len(losses))), y_value=np.array(losses),
                                            title="loss", x_label="iteration", y_label="loss"
                                        )
    plt.pause(1)
