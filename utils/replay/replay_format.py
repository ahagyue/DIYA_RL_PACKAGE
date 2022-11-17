import torch
from utils.replay.replayer_interface import ReplayInterface

def get_replay(batch:int, replay_buffer:ReplayInterface, device="cpu"):
    replay = replay_buffer.batch_replay(batch)
    prev_obs, action, reward, curr_obs, done = replay[0]
    for pobs, act, rew, cobs, don in replay[1:]:
        prev_obs += pobs
        action += act
        reward += rew
        curr_obs += cobs
        done += don

    prev_obs = torch.stack(prev_obs)
    action = torch.stack(act)
    reward = torch.stack(reward)
    curr_obs = torch.stack(curr_obs)
    done = torch.stack(done)
    
    return (
                prev_obs.type(torch.FloatTensor).to(device), action.to(device),
                reward.to(device), curr_obs.type(torch.FloatTensor).to(device),
                done.type(torch.FloatTensor).to(device)
            )