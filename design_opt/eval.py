import argparse
import os
import sys
sys.path.append(os.getcwd())

import yaml
from omegaconf import OmegaConf

from khrylib.utils import *
from design_opt.utils.config import Config
from design_opt.agents.genesis_agent import BodyGenAgent
from design_opt.utils.tools import set_global_seed

project_path = os.getcwd()

parser = argparse.ArgumentParser()
parser.add_argument('--restore_dir', type=str)
parser.add_argument('--epoch', default='best')
parser.add_argument('--save_video', action='store_true', default=False)
parser.add_argument('--pause_design', action='store_true', default=False)
args = parser.parse_args()

restore_dir = args.restore_dir

train_config_path = os.path.join(project_path, restore_dir, ".hydra", "config.yaml")

FLAGS = yaml.safe_load(open(train_config_path, 'r'))
FLAGS = OmegaConf.create(FLAGS)

cfg = Config(FLAGS, project_path, restore_dir)
cfg.restore_dir = restore_dir

dtype = torch.float64
torch.set_default_dtype(dtype)
device = torch.device('cpu')
set_global_seed(cfg.seed)

epoch = int(args.epoch) if isinstance(args.epoch, str) and args.epoch.isnumeric() else args.epoch

"""create agent"""
agent = BodyGenAgent(cfg=cfg, dtype=dtype, device=device, seed=cfg.seed, num_threads=1, training=True, checkpoint=epoch)

agent.visualize_agent(num_episode=1, mean_action=True, save_video=args.save_video, pause_design=True)
print("    design:", agent.env.robot.get_params(get_name=True), agent.env.robot.get_params(get_name=False))
