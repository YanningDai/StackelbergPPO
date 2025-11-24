# import argparse
import os
import sys
sys.path.append(os.getcwd())

from khrylib.utils import *
from design_opt.utils.config import Config
from design_opt.agents.genesis_agent import BodyGenAgent
from design_opt.utils.tools import set_global_seed
import wandb
import hydra
from omegaconf import DictConfig

project_path = os.getcwd()

def main_loop(FLAGS, job_dir):
    if FLAGS.render:
        FLAGS.num_threads = 1
        
    cfg = Config(FLAGS, project_path, job_dir)

    dtype = torch.float64
    torch.set_default_dtype(dtype)
    device = torch.device('cuda', index=FLAGS.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(FLAGS.gpu_index)
    set_global_seed(cfg.seed)

    start_epoch = int(FLAGS.epoch) if isinstance(FLAGS.epoch, str) and FLAGS.epoch.isnumeric() else FLAGS.epoch


    """create agent"""
    agent = BodyGenAgent(cfg=cfg, dtype=dtype, device=device, seed=cfg.seed, num_threads=FLAGS.num_threads, training=True, checkpoint=start_epoch)    

    if FLAGS.render:
        agent.pre_epoch_update(start_epoch)
        agent.sample(1e8, mean_action=not FLAGS.show_noise, render=True)
    else:
        for epoch in range(start_epoch, cfg.max_epoch_num):          
            agent.optimize(epoch)
            agent.save_checkpoint(epoch)

            """clean up gpu memory"""
            torch.cuda.empty_cache()

        agent.logger.info('training done!')


@hydra.main(version_base="1.2", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    FLAGS = cfg
    
    job_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    if FLAGS.enable_wandb:
        wandb.login()
        wandb.init(
            project=str(FLAGS.project),
            group=str(FLAGS.group),
            name=str(FLAGS.job_name),
            resume=False,
            dir=job_dir,
        )
    
    main_loop(FLAGS, job_dir)
    
    if FLAGS.enable_wandb:
        wandb.finish()
    
if __name__ == '__main__':
    main()
