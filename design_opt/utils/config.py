import yaml
import os
import glob
import numpy as np


class Config:

    def __init__(self, FLAG, project_path, job_dir):
        cfg_id = FLAG.cfg
        self.id = cfg_id
        self.project_path = project_path
        cfg_path = os.path.join(project_path, "design_opt", "cfg", f"{cfg_id}.yml")
        files = glob.glob(cfg_path, recursive=True)
        assert(len(files) == 1)
        cfg = yaml.safe_load(open(files[0], 'r'))
        # create dirs
        self.job_dir = job_dir 

        if not os.path.exists(self.job_dir):
            os.makedirs(self.job_dir)

        self.model_dir = '%s/models' % self.job_dir
        self.log_dir = '%s/log' % self.job_dir
        self.tb_dir = '%s/tb' % self.job_dir
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.tb_dir, exist_ok=True)
        
        restore_dir = FLAG.get('restore_dir', None)
        if restore_dir is not None:
            self.restore_dir = os.path.join(self.project_path, restore_dir)
            if not os.path.exists(self.restore_dir):
                raise FileNotFoundError(
                    f"Restore directory not found: {self.restore_dir}\n"
                    f"Please check the path and try again."
                )
        
        # training config
        self.epoch = FLAG.get('epoch', 0)
        self.gamma = cfg.get('gamma', 0.99)
        self.tau = cfg.get('tau', 0.95)
        self.agent_specs = cfg.get('agent_specs', dict())
        self.policy_specs = cfg.get('policy_specs', dict())
        self.policy_specs.update(FLAG.get('policy_specs', dict()))
        self.obs_specs = cfg.get('obs_specs', dict())
        self.obs_specs.update(FLAG.get('obs_specs', dict()))
        self.adv_clip = cfg.get('adv_clip', np.inf)
        self.eval_batch_size = FLAG.get('eval_batch_size', 10000)
        self.seed_method = cfg.get('seed_method', 'deep')
        
        # training config (from global flag)
        self.lr_decay = FLAG.get('lr_decay', False)
        self.policy_optimizer = FLAG.get('policy_optimizer', 'Adam')
        self.policy_lr = FLAG.get('policy_lr', 5e-5)
        self.policy_momentum = FLAG.get('policy_momentum', 0.0)
        self.policy_weightdecay = FLAG.get('policy_weightdecay', 0.0)
        self.value_specs = cfg.get('value_specs', dict())
        self.value_specs.update(FLAG.get('value_specs', dict()))
        self.value_optimizer = FLAG.get('value_optimizer', 'Adam')
        self.value_lr = FLAG.get('value_lr', 3e-4)
        self.value_momentum = FLAG.get('value_momentum', 0.0)
        self.value_weightdecay = FLAG.get('value_weightdecay', 0.0)
        self.clip_epsilon = FLAG.get('clip_epsilon', 0.2)
        self.num_optim_epoch = int(FLAG.get('num_optim_epoch', 10))
        self.min_batch_size = int(FLAG.get('min_batch_size', 50000))
        self.mini_batch_size = int(FLAG.get('mini_batch_size', self.min_batch_size))
        self.max_epoch_num = int(FLAG.get('max_epoch_num', 1000))
        self.seed = FLAG.get('seed', 0)
        self.save_model_interval = FLAG.get('save_model_interval', 100)
        self.norm_advantage = FLAG.get('norm_advantage', False)
        self.max_grad_norm = FLAG.get('max_grad_norm', 40)
        self.uni_obs_norm = FLAG.get('uni_obs_norm', False)
        self.norm_return = FLAG.get('norm_return', True)
        self.reward_shift = FLAG.get('reward_shift', 0.0)
        self.xml_name = FLAG.get('xml_name', 'default')
        self.planner_demean = FLAG.get('planner_demean', False)
        
        self.enable_wandb = FLAG.get('enable_wandb', True)
        self.group = FLAG.get('group', 'group')

        # anneal parameters
        self.scheduled_params = cfg.get('scheduled_params', dict())
        self.skel_entropy_coef = FLAG.get('skel_entropy_coef', 0.0)
        self.attr_entropy_coef = FLAG.get('attr_entropy_coef', 0.0)
        self.control_entropy_coef = FLAG.get('control_entropy_coef', 0.0)

        # env
        self.env_name = cfg.get('env_name', 'hopper')
        self.done_condition = cfg.get('done_condition', dict())
        self.env_specs = cfg.get('env_specs', dict())
        self.reward_specs = cfg.get('reward_specs', dict())
        self.add_body_condition = cfg.get('add_body_condition', dict())
        self.max_body_depth = cfg.get('max_body_depth', 4)
        self.min_body_depth = cfg.get('min_body_depth', 1)
        self.enable_remove = cfg.get('enable_remove', True)
        self.skel_transform_nsteps = FLAG.get('skel_transform_nsteps', 5)
        self.env_init_height = cfg.get('env_init_height', False)

        # robot config
        self.robot_param_scale = cfg.get('robot_param_scale', 0.1)
        self.robot_cfg = cfg.get('robot', dict())

        self.restore_dir = FLAG.get('restore_dir', None)
        self.skel_uniform_prob = FLAG.get('skel_uniform_prob', 0.0)
        self.stack_follower_steps = int(FLAG.get('stack_follower_steps', 6))
        self.lamda = FLAG.get('lamda', 1e-2)
        self.fisher_correct = FLAG.get('fisher_correct', False)
        self.stabilize_fisher = FLAG.get('stabilize_fisher', False)
        self.s_gradient_limit = FLAG.get('s_gradient_limit', -1)
        self.gradient_ratio_limit = FLAG.get('gradient_ratio_limit', -1)
        self.morph_prior = FLAG.get('morph_prior', False)
