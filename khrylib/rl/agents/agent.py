import multiprocessing
from khrylib.rl.core import LoggerRL, TrajBatch
from khrylib.utils.memory import Memory
from khrylib.utils.torch import *
import math
import time
import os
import platform
if platform.system() != "Linux":
    from multiprocessing import set_start_method
    set_start_method("fork")
os.environ["OMP_NUM_THREADS"] = "1"


class Agent:

    def __init__(self, env, policy_net, value_net, dtype, device, gamma, custom_reward=None,
                 end_reward=False, running_state=None, num_threads=1, logger_cls=LoggerRL, logger_kwargs=None, traj_cls=TrajBatch):
        self.env = env
        self.policy_net = policy_net
        self.value_net = value_net
        self.dtype = dtype
        self.device = device
        self.gamma = gamma
        self.custom_reward = custom_reward
        self.end_reward = end_reward
        self.running_state = running_state
        self.num_threads = num_threads
        self.noise_rate = 1.0
        self.traj_cls = traj_cls
        self.logger_cls = logger_cls
        self.logger_kwargs = dict() if logger_kwargs is None else logger_kwargs
        self.sample_modules = [policy_net]
        self.update_modules = [policy_net, value_net]