# from khrylib.rl.core import estimate_advantages
from khrylib.rl.agents.agent import Agent
from khrylib.utils.torch import *
import time


class AgentPG(Agent):

    def __init__(self, tau=0.95, optimizer_policy=None, optimizer_value=None,
                 opt_num_epochs=1, value_opt_niter=1, **kwargs):
        super().__init__(**kwargs)
        self.tau = tau
        self.optimizer_policy = optimizer_policy
        self.optimizer_value = optimizer_value
        self.opt_num_epochs = opt_num_epochs
        self.value_opt_niter = value_opt_niter
