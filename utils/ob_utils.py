import time
def log_time(prev_time=None, log='', return_time=False):
    if prev_time is not None :
        delta = time.time() - prev_time
        print("[TIME] ", log, delta)
    if return_time:
        return time.time(), delta
    else:
        return time.time()

from collections import defaultdict
import torch
import numpy as np
def _to_tensor(v):
    if torch.is_tensor(v):
        return v.type(torch.float)
    elif isinstance(v, np.ndarray):
        return torch.from_numpy(v).type(torch.float)
    else:
        return torch.tensor(v, dtype=torch.float)

def batch_obs(observations, device=None, need_list=[]):
    batch = defaultdict(list)
    for obs in observations:
        for sensor in obs:
            if isinstance(obs[sensor], dict): continue
            batch[sensor].append(_to_tensor(obs[sensor]))
    for sensor in batch:
        batch[sensor] = (
            torch.stack(batch[sensor], dim=0)
            .to(device=device)
            .to(dtype=torch.float)
        )

    return batch
