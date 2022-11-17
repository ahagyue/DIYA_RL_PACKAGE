import torch
from utils.replay.replayer_interface import ReplayInterface

def get_replay(batch:int, replay_buffer:ReplayInterface, device="cpu"):
    replay = replay_buffer.batch_replay(batch)
    prev_obs, action, reward, curr_obs, done = [], [], [], [], []
    for pobs, act, rew, cobs, don in replay:
        prev_obs.append(torch.tensor(pobs))
        action.append(torch.tensor([act]))
        reward.append(torch.tensor([rew]))
        curr_obs.append(torch.tensor(cobs))
        done.append(torch.tensor(don))

    prev_obs = torch.stack(prev_obs)
    action = torch.stack(action)
    reward = torch.stack(reward)
    curr_obs = torch.stack(curr_obs)
    done = torch.stack(done)
    
    return (
                prev_obs.type(torch.FloatTensor).to(device), action.to(device),
                reward.to(device), curr_obs.type(torch.FloatTensor).to(device),
                done.type(torch.FloatTensor).to(device)
            )