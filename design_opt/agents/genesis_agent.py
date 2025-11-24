import os
import math
import pickle
import time
from khrylib.utils import *
from khrylib.utils.torch import *
from khrylib.rl.agents import AgentPPO
from torch.utils.tensorboard import SummaryWriter
from design_opt.utils.stackelberg import *
from design_opt.envs import env_dict
from design_opt.models.bodygen_policy import BodyGenPolicy
from design_opt.models.bodygen_critic import BodyGenValue
from design_opt.utils.logger import LoggerRLV1
from design_opt.utils.tools import TrajBatchDisc, EpisodeBatchPlanner
import multiprocessing
from khrylib.rl.core.running_norm import RunningNorm
from torch.optim.lr_scheduler import LambdaLR
from design_opt.models.parameter_manager import ParameterManager
import wandb

        
def tensorfy(np_list, device=torch.device('cpu')):
    if isinstance(np_list[0], list):
        return [[torch.tensor(x).to(device) if i <= 1 or i == 4 or i >= 7 else x for i, x in enumerate(y)] for y in np_list]
    else:
        return [torch.tensor(y).to(device) for y in np_list]


class BodyGenAgent(AgentPPO):

    def __init__(self, cfg, dtype, device, seed, num_threads, training=True, checkpoint=0):
        self.cfg = cfg
        self.training = training
        self.device = device
        self.loss_iter = 0
        self.setup_env()
        self.env.seed(seed)
        self.setup_logger()
        self.setup_policy()
        self.setup_value()
        self.setup_optimizer()
        self.para_mgr = ParameterManager(self.cfg)
        if self.cfg.norm_return:
            self.design_ret_norm = RunningNorm(1, demean=self.cfg.planner_demean, clip=False)
            self.control_ret_norm = RunningNorm(1, demean=False, clip=False)
            self.ret_norm = RunningNorm(1, demean=False, clip=False)
        else:
            self.design_ret_norm = self.control_ret_norm = self.ret_norm = None
        if cfg.uni_obs_norm:
            self.obs_norm = RunningNorm(self.state_dim).to(self.device)
        else:
            self.obs_norm = None
        if checkpoint != 0:
            self.load_checkpoint(checkpoint)
        self.leader_step = self.cfg.skel_transform_nsteps + 1
        self.damping = self.cfg.lamda
        
        super().__init__(env=self.env, dtype=dtype, device=device, running_state=self.running_state,
                         custom_reward=None, logger_cls=LoggerRLV1, traj_cls=TrajBatchDisc, num_threads=num_threads,
                         policy_net=self.policy_net, value_net=self.value_net,
                         optimizer_policy=self.optimizer_policy, optimizer_value=self.optimizer_value, opt_num_epochs=cfg.num_optim_epoch,
                         gamma=cfg.gamma, tau=cfg.tau, clip_epsilon=cfg.clip_epsilon,
                         policy_grad_clip=[(self.policy_net.parameters(), 40)],
                         use_mini_batch=cfg.mini_batch_size < cfg.min_batch_size, mini_batch_size=cfg.mini_batch_size)

    ## Setting Ups        
    def setup_env(self):
        env_class = env_dict[self.cfg.env_name]
        self.env = env = env_class(self.cfg, self)
        self.attr_fixed_dim = env.attr_fixed_dim
        self.attr_design_dim = env.attr_design_dim
        self.sim_obs_dim = env.sim_obs_dim
        self.state_dim = self.attr_fixed_dim + self.sim_obs_dim + self.attr_design_dim
        self.control_action_dim = env.control_action_dim
        self.skel_num_action = env.skel_num_action
        self.action_dim = self.control_action_dim + self.attr_design_dim
        self.running_state = None
        
    def setup_logger(self):
        cfg = self.cfg
        self.tb_logger = SummaryWriter(cfg.tb_dir) if self.training else None
        self.logger = create_logger(os.path.join(cfg.log_dir, f'log_{"train" if self.training else "eval"}.txt'), file_handle=True)
        self.best_rewards = -1000.0
        self.save_best_flag = False
        
    def setup_policy(self):
        cfg = self.cfg
        self.policy_net = BodyGenPolicy(cfg.policy_specs, self)
        to_device(self.device, self.policy_net)
        self.ep_planner = EpisodeBatchPlanner(m=self.cfg.stack_follower_steps, n=self.cfg.num_optim_epoch, pad_value=-1, shuffle_episode=True)
        
    def setup_value(self):
        cfg = self.cfg
        self.value_net = BodyGenValue(cfg.value_specs, self)
        to_device(self.device, self.value_net)
        
    def setup_optimizer(self):
        cfg = self.cfg
        # policy optimizer
        if cfg.policy_optimizer == 'Adam':
            self.optimizer_policy = torch.optim.Adam(self.policy_net.parameters(), lr=cfg.policy_lr, weight_decay=cfg.policy_weightdecay)
        else:
            self.optimizer_policy = torch.optim.SGD(self.policy_net.parameters(), lr=cfg.policy_lr, momentum=cfg.policy_momentum, weight_decay=cfg.policy_weightdecay)
        # value optimizer
        if cfg.value_optimizer == 'Adam':
            self.optimizer_value = torch.optim.Adam(self.value_net.parameters(), lr=cfg.value_lr, weight_decay=cfg.value_weightdecay)
        else:
            self.optimizer_value = torch.optim.SGD(self.value_net.parameters(), lr=cfg.value_lr, momentum=cfg.value_momentum, weight_decay=cfg.value_weightdecay)
        # learning rate decay
        if self.cfg.lr_decay:
            self.scheduler_policy = LambdaLR(self.optimizer_policy, lr_lambda=lambda epoch: 1 - epoch / self.cfg.max_epoch_num)
            self.scheduler_value = LambdaLR(self.optimizer_value, lr_lambda=lambda epoch: 1 - epoch / self.cfg.max_epoch_num)
        else:
            self.scheduler_policy = None
            self.scheduler_value = None

    ## Sampling
    def sample(self, min_batch_size, mean_action=False, render=False, nthreads=None):
        if nthreads is None:
            nthreads = self.num_threads
        t_start = time.time()

        to_test(*self.sample_modules)
        if self.cfg.uni_obs_norm:
            self.obs_norm.eval()
            self.obs_norm.to('cpu')
        
        with to_cpu(*self.sample_modules):
            with torch.no_grad():
                thread_batch_size = int(math.floor(min_batch_size / nthreads)) 
                queue = multiprocessing.Queue()
                memories = [None] * nthreads
                loggers = [None] * nthreads
                for i in range(nthreads-1):
                    worker_args = (i+1, queue, thread_batch_size, mean_action, render)
                    worker = multiprocessing.Process(target=self.sample_worker, args=worker_args)
                    worker.start()
                memories[0], loggers[0] = self.sample_worker(0, None, thread_batch_size, mean_action, render)

                for i in range(nthreads - 1):
                    pid, worker_memory, worker_logger = queue.get()
                    memories[pid] = worker_memory
                    loggers[pid] = worker_logger
                traj_batch = self.traj_cls(memories)
                logger = self.logger_cls.merge(loggers, **self.logger_kwargs)

        logger.sample_time = time.time() - t_start
        return traj_batch, logger

    ## Per worker sampling
    def sample_worker(self, pid, queue, min_batch_size, mean_action, render):
        ## make seed for the worker
        if pid > 0:
            torch.manual_seed(torch.randint(0, 5000, (1,)) * pid)
            if hasattr(self.env, 'np_random'):
                self.env.np_random.seed(self.env.np_random.randint(5000) * pid)
        
        memory = Memory()
        logger = self.logger_cls(**self.logger_kwargs)

        while logger.num_steps < min_batch_size:
            state = self.env.reset()
            logger.start_episode(self.env)
            episode_reward = 0.0
            while True:
                state_var = tensorfy([state])
                
                ## do obs norm (none-updated)
                if self.cfg.uni_obs_norm:
                    state_var = self.normalize_observation(state_var)

                use_mean_action = mean_action or torch.bernoulli(torch.tensor([1 - self.noise_rate])).item()
                action = self.policy_net.select_action(state_var, use_mean_action).numpy().astype(np.float64)
                next_state, env_reward, termination, truncation, info = self.env.step(action)
                reward = env_reward
                c_reward = info.get('reward_ctrl', 0)
                # add end reward
                if self.end_reward and info.get('end', False):
                    reward += self.env.end_reward
                    
                if info['stage'] == 'execution':
                    reward += self.cfg.reward_shift 
                
                # logging
                logger.step(self.env, env_reward, c_reward, 0.0, info)

                done = (termination or truncation)
                exp = 1 - use_mean_action
                
                memory.push(state, action, termination, done, next_state, reward, exp, c_reward)
                episode_reward += reward
                if done:
                    episode_reward = 0.0
                    break
                state = next_state
                
            logger.end_episode(self.env)        
        logger.end_sampling()

        if queue is not None:
            queue.put([pid, memory, logger])
        else:
            return memory, logger

    def optimize(self, epoch):
        info = self.optimize_policy(epoch)
        if self.scheduler_policy is not None:
            self.scheduler_policy.step()
        if self.scheduler_value is not None:
            self.scheduler_value.step()
        self.log_optimize_policy(epoch, info)

    def optimize_policy(self, epoch):

        """generate multiple trajectories that reach the minimum batch_size"""
        t0 = time.time()
        batch, log = self.sample(self.cfg.min_batch_size)

        """update networks"""
        t1 = time.time()

        log_stackel = self.update_params(batch)
        t2 = time.time()

        """evaluate policy"""
        _, log_eval = self.sample(self.cfg.eval_batch_size, mean_action=True)
        print("Evaluation: ", [self.env.robot.bodies[i] for i in range(len(self.env.robot.bodies))])
        t3 = time.time()

        info = {
            'log': log, 'log_eval': log_eval, 'log_stackel': log_stackel, 'T_sample': t1 - t0, 'T_update': t2 - t1, 'T_eval': t3 - t2, 'T_total': t3 - t0
        }
        return info
        
    def estimate_advantages(self, states, next_states, rewards, next_terminations, next_dones, state_types, next_state_types, c_reward):
        design_masks = (state_types!=2).bool().to(self.device)
        control_masks = (state_types==2).bool().to(self.device)
        next_design_masks = (next_state_types!=2).bool().to(self.device)
        next_control_masks = (next_state_types==2).bool().to(self.device)
        self.design_ret_norm.to(self.device)
        self.control_ret_norm.to(self.device)
        
        with to_test(*self.update_modules):
            with torch.no_grad():
                values = []
                chunk = 10000
                for i in range(0, len(states), chunk):
                    states_i = states[i:min(i + chunk, len(states))]
                    values_i = self.value_net(states_i)
                    current_range = np.arange(i, min(i + chunk, len(states)))
                    if self.design_ret_norm is not None:
                        local_design_masks = design_masks[current_range]
                        values_i[local_design_masks] = self.design_ret_norm.unscale(values_i[local_design_masks])
                    if self.control_ret_norm is not None:
                        local_control_masks = control_masks[current_range]
                        values_i[local_control_masks] = self.control_ret_norm.unscale(values_i[local_control_masks])
                        
                    values.append(values_i)
                values = torch.cat(values)
                
                next_values = torch.zeros_like(values)
                next_values[:-1] = values[1:]
                
                indices = torch.where(next_dones)[0]
                compute_next_states = [next_states[i] for i in indices]
                if compute_next_states:
                    computed_values = self.value_net(compute_next_states)
                    if self.design_ret_norm is not None:
                        local_next_design_masks = next_design_masks[indices]
                        computed_values[local_next_design_masks] = self.design_ret_norm.unscale(computed_values[local_next_design_masks])
                    if self.control_ret_norm is not None:
                        local_next_control_masks = next_control_masks[indices]
                        computed_values[local_next_control_masks] = self.control_ret_norm.unscale(computed_values[local_next_control_masks])
                        
                    next_values[indices] = computed_values
                        
        self.design_ret_norm.to('cpu')
        self.control_ret_norm.to('cpu')
        
        device = rewards.device
        rewards, next_terminations, next_dones, values, next_values = batch_to(torch.device('cpu'), rewards, next_terminations, next_dones, values, next_values)
        c_reward, = batch_to(torch.device('cpu'), c_reward)
        design_masks, control_masks = batch_to(torch.device('cpu'), design_masks, control_masks)
        
        tensor_type = type(rewards)
        deltas = tensor_type(rewards.size(0), 1)
        advantages = tensor_type(rewards.size(0), 1)
        design_returns = tensor_type(rewards.size(0), 1)

        next_advantage = 0
        next_design_return = 0
        for i in reversed(range(rewards.size(0))):
            deltas[i] = rewards[i] + self.gamma * next_values[i] * (1 - next_terminations[i]) - values[i]
            advantages[i] = next_advantage = deltas[i] + self.gamma * self.tau * next_advantage * (1 - next_dones[i])
            design_returns[i] = next_design_return = rewards[i] + next_design_return * (1 - next_dones[i])
            
        design_advantages = design_returns - values
        returns = values + advantages
        
        if self.design_ret_norm is not None:
            returns[design_masks] = self.design_ret_norm(design_returns[design_masks])
        else:
            returns[design_masks] = design_returns[design_masks]
            
        if self.control_ret_norm is not None:
            returns[control_masks] = self.control_ret_norm(returns[control_masks])
        
        if self.cfg.norm_advantage:
            advantages[design_masks] = (design_advantages[design_masks] - design_advantages[design_masks].mean()) / design_advantages[design_masks].std()
            advantages[control_masks] = (advantages[control_masks] - advantages[control_masks].mean()) / advantages[control_masks].std()

        advantages, returns = batch_to(device, advantages, returns)
        return advantages, returns

    def update_params(self, batch, update_list=None):
        t0 = time.time()
        
        to_train(*self.update_modules)
        
        states = tensorfy(batch.states, self.device)
        next_states = tensorfy(batch.next_states, self.device)
        actions = tensorfy(batch.actions, self.device)
        rewards = torch.from_numpy(batch.rewards).to(self.dtype).to(self.device)
        next_terminations = torch.from_numpy(batch.next_terminations).to(self.dtype).to(self.device)
        next_dones = torch.from_numpy(batch.next_dones).to(self.dtype).to(self.device)
        exps = torch.from_numpy(batch.exps).to(self.dtype).to(self.device)
        c_reward = torch.from_numpy(batch.c_reward).to(self.dtype).to(self.device)
        
        if self.cfg.uni_obs_norm:
            self.obs_norm.to(self.device)
            self.obs_norm.train()
            states = self.normalize_observation(states)
            self.obs_norm.eval()
            next_states = self.normalize_observation(next_states)

        log_stakel = self.update_policy(states, next_states, rewards, next_terminations, next_dones, actions, exps, c_reward, update_list)

        return log_stakel
    
    def normalize_observation(self, x):
        obs, edges, use_transform_action, num_nodes, body_ind, body_depths, body_heights, distances, lapPE = zip(*x)
        obs_cat = torch.cat(obs)
        obs_norm = self.obs_norm(obs_cat)
        indices = np.cumsum(num_nodes)
        obs_split = [obs_norm[start:end] for start, end in zip([0] + list(indices[:-1]), indices)]
        x = [list(item) for item in zip(obs_split, edges, use_transform_action, num_nodes, body_ind, body_depths, body_heights, distances, lapPE)]
        return x

    def get_perm_batch_design(self, states):
        inds = [[], [], []]
        for i, x in enumerate(states):
            use_transform_action = x[2]
            inds[use_transform_action.item()].append(i)
        perm = np.array(inds[0] + inds[1] + inds[2])
        return perm, LongTensor(perm).to(self.device)

    def get_perm_follower_design(self, states):
        inds = []
        for i, x in enumerate(states):
            use_transform_action = x[2]
            if use_transform_action == 2:
                inds.append(i)
        perm = np.array(inds)
        return perm, LongTensor(perm).to(self.device)
    
    def update_policy(self, states, next_states, rewards, next_terminations, next_dones, actions, exps, c_reward, update_list=None):
        """update policy"""
        with to_test(*self.update_modules):
            with torch.no_grad():
                fixed_log_probs = []
                chunk = 10000
                for i in range(0, len(states), chunk):
                    states_i = states[i:min(i + chunk, len(states))]
                    actions_i = actions[i:min(i + chunk, len(states))]
                    fixed_log_probs_i = self.policy_net.get_log_prob(states_i, actions_i)
                    fixed_log_probs.append(fixed_log_probs_i)
                fixed_log_probs = torch.cat(fixed_log_probs)
        num_state = len(states)
        
        state_types = torch.from_numpy(np.array([item[2] for item in states])).to(torch.int)
        next_state_types = torch.from_numpy(np.array([item[2] for item in next_states])).to(torch.int)
        
        advantages, returns = self.estimate_advantages(states, next_states, rewards, next_terminations, next_dones, state_types, next_state_types, c_reward)

        # Assign the corresponding first m follower steps to each leader
        episodes = self.split_episodes_from_dones(next_dones)  # List[np.ndarray]
        state_types_np = state_types.detach().cpu().numpy().reshape(-1).astype(np.int64)
        leader_idx_ep, follower_idx_ep = self.ep_planner.build_episode_indices(episodes, state_types_np)
        valid = [i for i, (L, F) in enumerate(zip(leader_idx_ep, follower_idx_ep)) if (len(L) >= self.leader_step) and (len(F) > 0)]

        dropped = len(episodes) - len(valid)
        if dropped > 0:
            episodes        = [episodes[i]        for i in valid]
            leader_idx_ep   = [leader_idx_ep[i]   for i in valid]
            follower_idx_ep = [follower_idx_ep[i] for i in valid]
            print(f"[EP-FILTER] dropped {dropped} episodes "
                f"(rule: leader_len >= {self.leader_step}, follower>0)")
            assert len(episodes) > 0, "All episodes are filtered"

        Fmat = self.ep_planner.make_Fmat(follower_idx_ep)
        
        # Update epoch counts
        E = len(episodes)
        MB = self.mini_batch_size if self.use_mini_batch else self.cfg.min_batch_size
        m = self.ep_planner.m
        E_per_batch = int(np.ceil(MB / self.leader_step))
        leader_optim_num = int(np.ceil(E / E_per_batch))

        if update_list is not None and ('skel_trans' not in update_list) and ('attr_trans' not in update_list):
            leader_optim_num = 0
        
        use_all_follower_for_khv = False # todo: may use all follower samples for khv
        for k in range(self.opt_num_epochs):
            
            # ---------- Leader: preserve episode order and use Stackelberg gradient during update ----------
            fol_slice_all = Fmat[:, k*m:(k+1)*m]
            ep_order = np.random.permutation(E)

            for j in range(leader_optim_num):
                ep_sel = ep_order[j*E_per_batch : (j+1)*E_per_batch]
                li = [leader_idx_ep[e] for e in ep_sel if leader_idx_ep[e].size > 0]
                assert len(li) > 0, "No valid leader indices found"
                
                leader_idx_np = np.concatenate(li, axis=0)
                follower_idx_np = fol_slice_all[ep_sel, :].ravel()
                leader_idx = LongTensor(leader_idx_np).to(self.device)
                follower_idx = LongTensor(follower_idx_np).to(self.device)
                
                rndL_states, rndL_actions, rndL_adv, rndL_ret, rndL_fixlog, rndL_exps = \
                    index_select_list(states, leader_idx_np), index_select_list(actions, leader_idx_np), advantages[leader_idx].clone(), \
                    returns[leader_idx].clone(), fixed_log_probs[leader_idx].clone(), exps[leader_idx].clone()

                rndF_states, rndF_actions, rndF_adv, rndF_ret, rndF_fixlog, rndF_exps = \
                    index_select_list(states, follower_idx_np), index_select_list(actions, follower_idx_np), advantages[follower_idx].clone(), \
                    returns[follower_idx].clone(), fixed_log_probs[follower_idx].clone(), exps[follower_idx].clone()
                
                self.update_value(rndL_states, rndL_ret, update_list=['skel_trans', 'attr_trans'])
                surr_loss, Ja, Jc, log_stackel = self.ppo_loss_stackelberg(rndL_states, rndL_actions, rndL_adv, rndL_fixlog, rndF_states, rndF_actions, rndF_adv, rndF_fixlog, flag_log=(leader_optim_num-1==j))
                
                # ----------------- Implicit gradient computation -----------------
                theta1_params = self.para_mgr.get_param_tensors(self.policy_net, ['skel_trans', 'attr_trans'])
                theta2_params = self.para_mgr.get_param_tensors(self.policy_net, ['execution'])
                
                def policy_exec(states_batch):
                    out = self.policy_net(states_batch)
                    if isinstance(out, tuple):
                        out = out[0]
                    assert isinstance(out, torch.distributions.Distribution), f"policy_exec must return a Distribution, got {type(out)}"
                    return out
    
                J_delta = bilevel_leader_grad_correct(
                    J_a=Ja,     # K 
                    J_c=Jc,     # v 
                    theta1_list=theta1_params,
                    theta2_list=theta2_params,
                    policy_exec=policy_exec,
                    F_states=rndF_states,
                    damping=self.damping, cg_max_iter=20, cg_tol=1e-3, verbose=False, fisher_correct = self.cfg.fisher_correct
                )
                
                # print(f"Original surr_loss: {surr_loss:.6e}    delta J: {J_delta:.6e}")

                # --------- Gradient scaling and correction ---------
                # Compute gradients
                g_surr = torch.autograd.grad(surr_loss, theta1_params, retain_graph=True, allow_unused=True)
                g_surr = [g.detach() if g is not None else torch.zeros_like(p) for g, p in zip(g_surr, theta1_params)]

                g_delta = torch.autograd.grad(J_delta, theta1_params, retain_graph=False, allow_unused=True)
                g_delta = [g.detach() if g is not None else torch.zeros_like(p) for g, p in zip(g_delta, theta1_params)]
                
                def flat(vs): return torch.cat([v.reshape(-1) for v in vs])
                gs, gd = flat(g_surr), flat(g_delta)
                
                # Inspect and correct gradients
                cos = torch.dot(gs, gd) / (gs.norm() * gd.norm() + 1e-12)
                ratio = gd.norm() / (gs.norm() + 1e-12)
                # print(f"[Probe θ1] ||g_surr||={gs.norm():.3e}  ||g_delta||={gd.norm():.3e}  cos={cos:.3f}  ratio={ratio:.3f}")
                
                # method1
                ratio_limit = self.cfg.gradient_ratio_limit
                if ratio_limit > 0.0 and ratio > ratio_limit: # scale down g_delta if ratio exceeds limit
                    scale = ratio_limit / (ratio + 1e-12) 
                    g_delta = [gd * scale for gd in g_delta] 
                    # print(f" [Clamp] ratio > limit={ratio_limit:.3f}, scale g_delta by {scale:.3f}") 
                    
                # method2
                clip_val = self.cfg.s_gradient_limit  # 0.05~0.2
                if clip_val > 0.0:
                    g_delta = [torch.clamp(gd, min=-clip_val, max=clip_val) if gd is not None else None for gd in g_delta]
                    print(f"   [Clip] elementwise clamp g_delta to ±{clip_val:.3g}")
                
                g_corrected = [gs - gd for gs, gd in zip(g_surr, g_delta)]
                        
                # ---------- Update ----------
                self.optimizer_policy.zero_grad()
                
                for p, g in zip(theta1_params, g_corrected):  # Assign corrected gradients back; no backward needed
                    if g is not None:
                        p.grad = g.clone()
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.cfg.max_grad_norm)
                self.optimizer_policy.step()
            
            # -------------------------- Follower: normal update ----------------------------
            if update_list is not None and ('execution' not in update_list):
                continue
            perm_np, _ = self.get_perm_follower_design(states) 
            perm_np = np.random.permutation(perm_np)
            perm = LongTensor(perm_np).to(self.device)

            rnd_states, rnd_actions, rnd_returns, rnd_advantages, rnd_fixed_log_probs, rnd_exps = \
                index_select_list(states, perm_np), index_select_list(actions, perm_np), returns[perm].clone(), advantages[perm].clone(), \
                fixed_log_probs[perm].clone(), exps[perm].clone()

            follower_optim_num = int(math.floor(len(perm_np) / MB)) if len(perm_np) > 0 else 0
            
            for i in range(follower_optim_num):
                ind = slice(i * self.mini_batch_size, min((i + 1) * self.mini_batch_size, num_state))
                states_b, actions_b, advantages_b, returns_b, fixed_log_probs_b, exps_b = \
                    rnd_states[ind], rnd_actions[ind], rnd_advantages[ind], rnd_returns[ind], rnd_fixed_log_probs[ind], rnd_exps[ind]
                
                self.update_value(states_b, returns_b, update_list=['execution'])
                surr_loss = self.ppo_loss(states_b, actions_b, advantages_b, fixed_log_probs_b)
                self.optimizer_policy.zero_grad()
                self.para_mgr.selective_backward(surr_loss, self.policy_net, update_list=['execution'])

                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.cfg.max_grad_norm)
                self.optimizer_policy.step()
                
        return log_stackel
    
    def split_episodes_from_dones(self, next_dones_t: torch.Tensor):
        dones = next_dones_t.detach().cpu().numpy().astype(bool).reshape(-1)
        ends = np.where(dones)[0]
        if len(ends) == 0:
            return [np.arange(0, len(dones), dtype=np.int64)]
        starts = np.r_[0, ends[:-1] + 1]
        return [np.arange(s, e + 1, dtype=np.int64) for s, e in zip(starts, ends)]
    
    def update_value(self, states, returns, update_list=None):
        """update critic"""
        for _ in range(self.value_opt_niter):
            values_pred = self.value_net(states)
            value_loss = (values_pred - returns).pow(2).mean()
            self.optimizer_value.zero_grad()
            if hasattr(self, "para_mgr"):
                self.para_mgr.selective_backward(value_loss, self.value_net, update_list)
            else:
                value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), self.cfg.max_grad_norm)
            self.optimizer_value.step()

    def ppo_loss_stackelberg(self, L_states, L_actions, L_adv, L_fixlog, F_states, F_actions, F_adv, F_fixlog, flag_log=False):
        L_log_probs = self.policy_net.get_log_prob(L_states, L_actions)
        F_log_probs = self.policy_net.get_log_prob(F_states, F_actions)

        L_ratio = torch.exp(L_log_probs - L_fixlog) # pi/pi_old
        F_ratio = torch.exp(F_log_probs - F_fixlog) 

        # leader ppo loss
        L_surr1 = L_ratio * L_adv
        L_surr2 = torch.clamp(L_ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * L_adv
        L_surr_loss = - torch.min(L_surr1, L_surr2).mean()

        log = {}
        if flag_log:
            with torch.no_grad():
                L_log_ratio = L_log_probs - L_fixlog   # = log pi_new - log pi_old
                F_log_ratio = F_log_probs - F_fixlog
                
                # kl
                L_approx_kl = (-L_log_ratio).mean()
                F_approx_kl = (-F_log_ratio).mean()
                
                # kl3
                L_kl3 = (L_ratio - 1 - L_log_ratio).mean()
                F_kl3 = (F_ratio - 1 - F_log_ratio).mean()

                # fraction of PPO ratios outside clipping bounds
                L_clip_frac = ((L_ratio > 1 + self.clip_epsilon) | (L_ratio < 1 - self.clip_epsilon)).float().mean()
                F_clip_frac = ((F_ratio > 1 + self.clip_epsilon) | (F_ratio < 1 - self.clip_epsilon)).float().mean()

                
                log = {
                    'L_surr_loss': L_surr_loss.item(),
                    'L_approx_kl': L_approx_kl.item(),
                    'F_approx_kl': F_approx_kl.item(),
                    'L_kl3': L_kl3.item(),
                    'F_kl3': F_kl3.item(),
                    'L_clip_frac': L_clip_frac.item(),
                    'F_clip_frac': F_clip_frac.item(),
                }
        
        # Jc
        F_surr1 = F_ratio * F_adv
        F_surr2 = torch.clamp(F_ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * F_adv
        Jc = -torch.min(F_surr1, F_surr2).mean() 

        # Ja
        E = L_ratio.numel() // self.leader_step # number of episodes
        assert L_ratio.numel() % self.leader_step == 0, f"L_ratio size {L_ratio.numel()} not divisible by leader_step={self.leader_step}"
        
        m = self.ep_planner.m  # number of follower steps per episode
        assert F_ratio.numel() % m == 0 and F_adv.numel() % m == 0, f"Follower tensors not multiple of m={m}: F_ratio={F_ratio.numel()}, F_adv={F_adv.numel()}"

        L_ratio = L_ratio.view(E, self.leader_step)
        l_bar = L_ratio.mean(dim=1)                  # [E]
        F_ratio = F_ratio.view(E, m)
        F_adv   = F_adv.view(E, m)
        f_bar = (F_ratio * F_adv).mean(dim=1)        # [E]
        Ja = -(l_bar * f_bar).mean()

        return L_surr_loss, Ja, Jc, log

    def ppo_loss(self, states, actions, advantages, fixed_log_probs):
        log_probs = self.policy_net.get_log_prob(states, actions)
        ratio = torch.exp(log_probs - fixed_log_probs) # pi/pi_old
        advantages = advantages
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
        surr_loss_step = - torch.min(surr1, surr2)
        surr_loss = surr_loss_step.mean()
        return surr_loss
                            
    def load_checkpoint(self, checkpoint):
        cfg = self.cfg
        if isinstance(checkpoint, int):
            cp_path = '%s/epoch_%04d.p' % (os.path.join(cfg.project_path,cfg.restore_dir,"models"), checkpoint)
            epoch = checkpoint
        else:
            assert isinstance(checkpoint, str)
            cp_path = '%s/%s.p' % (os.path.join(cfg.project_path,cfg.restore_dir,"models"), checkpoint)
        self.logger.info('loading model from checkpoint: %s' % cp_path)
        model_cp = pickle.load(open(cp_path, "rb"))
        
        new_policy_dict = {}
        for key, value in model_cp['policy_dict'].items():
            new_key = key
            if ".lin_r." in new_key:
                new_key = new_key.replace(".lin_r.", ".lin_root.")
            if  ".lin_l." in new_key:
                new_key = new_key.replace(".lin_l.", ".lin_rel.")
            new_policy_dict[new_key] = value

        new_value_dict = {}
        for key, value in model_cp['value_dict'].items():
            new_key = key
            if ".lin_r." in new_key:
                new_key = new_key.replace(".lin_r.", ".lin_root.")
            if  ".lin_l." in new_key:
                new_key = new_key.replace(".lin_l.", ".lin_rel.")
            new_value_dict[new_key] = value
        
        if_strict = True
        if cfg.morph_prior:
            prefix_include = ("skel_", "attr_")
            new_policy_dict = self.filter_keys(new_policy_dict, prefix_include=prefix_include)
            new_value_dict = self.filter_keys(new_value_dict,  prefix_include=prefix_include)
            if_strict = False

        self.policy_net.load_state_dict(new_policy_dict, strict=if_strict)
        self.value_net.load_state_dict(new_value_dict, strict=if_strict)
        
        self.loss_iter = model_cp['loss_iter']
        self.best_rewards = model_cp.get('best_rewards', self.best_rewards)
        if model_cp['obs_norm'] is not None and cfg.uni_obs_norm and not cfg.morph_prior:
            self.obs_norm.load_state_dict(model_cp['obs_norm'])
    
    def filter_keys(self, sd, prefix_exclude=None, prefix_include=None):
        if prefix_exclude is not None:
            return {k: v for k, v in sd.items() if not k.startswith(prefix_exclude)}
        if prefix_include is not None:
            return {k: v for k, v in sd.items() if k.startswith(prefix_include)}
        return sd
    
    def save_checkpoint(self, epoch):

        def save(cp_path):
            with to_cpu(self.policy_net, self.value_net):
                model_cp = {
                            'obs_norm': self.obs_norm.state_dict() if self.obs_norm is not None else None,
                            'policy_dict': self.policy_net.state_dict(),
                            'value_dict': self.value_net.state_dict(),
                            'loss_iter': self.loss_iter,
                            'best_rewards': self.best_rewards,
                            'epoch': epoch}
                pickle.dump(model_cp, open(cp_path, 'wb'))

        cfg = self.cfg
        additional_saves = self.cfg.agent_specs.get('additional_saves', None)
        if (cfg.save_model_interval > 0 and (epoch+1) % cfg.save_model_interval == 0) or \
           (additional_saves is not None and (epoch+1) % additional_saves[0] == 0 and epoch+1 <= additional_saves[1]):
            self.tb_logger.flush()
            save('%s/epoch_%04d.p' % (cfg.model_dir, epoch + 1))
        if self.save_best_flag:
            self.tb_logger.flush()
            self.logger.info(f'save best checkpoint with rewards {self.best_rewards:.2f}!')
            save('%s/best.p' % cfg.model_dir)

    def log_optimize_policy(self, epoch, info):
        cfg = self.cfg
        log, log_eval,log_stackel = info['log'], info['log_eval'], info['log_stackel']
        logger, tb_logger = self.logger, self.tb_logger
        log_str = f'{epoch}\tT_sample {info["T_sample"]:.2f}\tT_update {info["T_update"]:.2f}\tT_eval {info["T_eval"]:.2f}\t'\
            f'ETA {get_eta_str(epoch, cfg.max_epoch_num, info["T_total"])}\ttrain_R {log.avg_reward:.2f}\ttrain_R_eps {log.avg_episode_reward:.2f}\t'\
            f'exec_R {log_eval.avg_exec_reward:.2f}\texec_R_eps {log_eval.avg_exec_episode_reward:.2f}\t{cfg.id}'
        logger.info(log_str)

        if log_eval.avg_exec_episode_reward > self.best_rewards:
            self.best_rewards = log_eval.avg_exec_episode_reward
            self.save_best_flag = True
        else:
            self.save_best_flag = False

        tb_logger.add_scalar('train_R_avg ', log.avg_reward, epoch)
        tb_logger.add_scalar('policy_learning_rate', self.optimizer_policy.param_groups[0]["lr"], epoch)
        tb_logger.add_scalar('value_learning_rate', self.optimizer_value.param_groups[0]["lr"], epoch)
        tb_logger.add_scalar('train_R_eps_avg', log.avg_episode_reward, epoch)
        tb_logger.add_scalar('eval_R_eps_avg', log_eval.avg_episode_reward, epoch)
        tb_logger.add_scalar('exec_R_avg', log_eval.avg_exec_reward, epoch)
        tb_logger.add_scalar('exec_R_eps_avg', log_eval.avg_exec_episode_reward, epoch)
        tb_logger.add_scalar('reward_shift', self.cfg.reward_shift, epoch)
        
        if self.cfg.enable_wandb:
            wandb.log({
                'train_R_avg': log.avg_reward,
                'policy_learning_rate': self.optimizer_policy.param_groups[0]["lr"],
                'value_learning_rate': self.optimizer_value.param_groups[0]["lr"],
                'train_R_eps_avg': log.avg_episode_reward,
                'eval_R_eps_avg': log_eval.avg_episode_reward,
                'exec_R_avg': log_eval.avg_exec_reward,
                'exec_R_eps_avg': log_eval.avg_exec_episode_reward,
                'reward_shift': self.cfg.reward_shift,
                'avg_episode_len': log_eval.avg_episode_len,
                'episode_c_reward': log_eval.avg_episode_c_reward,
                'stackelberg/L_surr_loss': log_stackel["L_surr_loss"],
                'stackelberg/L_approx_kl': log_stackel["L_approx_kl"],
                'stackelberg/F_approx_kl': log_stackel["F_approx_kl"],
                'stackelberg/L_kl3': log_stackel["L_kl3"],
                'stackelberg/F_kl3': log_stackel["F_kl3"],
                'stackelberg/L_clip_frac': log_stackel["L_clip_frac"],
                'stackelberg/F_clip_frac': log_stackel["F_clip_frac"],
                
            }, step = epoch * self.cfg.min_batch_size)

    def visualize_agent(self, num_episode=1, mean_action=True, save_video=False, pause_design=True, max_num_frames=1000):
        import datetime
        env = self.env
        env.model.opt.timestep = 0.001
        paused = not save_video and pause_design
        
        import subprocess, numpy as np
        from OpenGL import GL
        import glfw

        ffmpeg = None           
        video_path = None
        FPS = 45               
        RENDER_EVERY = 1       
        
        if self.cfg.uni_obs_norm:
            self.obs_norm.eval()
            self.obs_norm.to('cpu')
        for i in range(num_episode):
            state = env.reset()

            print(f"Episode {i+1}/{num_episode}")
            episode_reward = 0.0
            custom_reward = 0.0
            
            for t in range(10000):
                state_var = tensorfy([state])
                if t == 0:
                    print(f"initial bodies = {[self.env.robot.bodies[i] for i in range(len(self.env.robot.bodies))]}")

                if self.cfg.uni_obs_norm:
                    state_var = self.normalize_observation(state_var)

                with torch.no_grad():
                    action = self.policy_net.select_action(state_var, mean_action).numpy().astype(np.float64)

                next_state, env_reward, termination, truncation, info = env.step(action)
                done = (termination or truncation)
                episode_reward += env_reward
                custom_reward += info.get('reward_ctrl', 0.0)

                if t < 5:
                    print(f"Step {t}: skel action = {action[:, -1]}, bodies = {[self.env.robot.bodies[i] for i in range(len(self.env.robot.bodies))]}")

                if t == self.cfg.skel_transform_nsteps:
                    env._get_viewer('human')._paused = paused
                
                if t > self.cfg.skel_transform_nsteps:
                    if t == self.cfg.skel_transform_nsteps + 1:
                        viewer = env._get_viewer('human')
                        viewer._paused = paused
                        if save_video:
                            viewer._hide_overlay = True                    
                    else:
                        env._get_viewer('human')._paused = False
                    
                    env.render()
                    if save_video and (t % RENDER_EVERY == 0):
                        glfw.make_context_current(viewer.window)
                        fb_w, fb_h = glfw.get_framebuffer_size(viewer.window)
                        GL.glFinish()
                        data = GL.glReadPixels(0, 0, fb_w, fb_h, GL.GL_RGB, GL.GL_UNSIGNED_BYTE)
                        frame = np.frombuffer(data, dtype=np.uint8).reshape(fb_h, fb_w, 3)
                        frame = frame[::-1]  

                        if ffmpeg is None:
                            import os
                            os.makedirs("out/videos", exist_ok=True)
                            video_path = f"out/videos/{self.cfg.id}_seed={self.cfg.seed}.mp4"
                            ffmpeg = subprocess.Popen([
                                "ffmpeg", "-y",
                                "-f", "rawvideo", "-vcodec", "rawvideo",
                                "-pix_fmt", "rgb24",
                                "-s", f"{fb_w}x{fb_h}",
                                "-r", str(FPS),
                                "-i", "-",
                                "-an",
                                "-vcodec", "libx264", "-pix_fmt", "yuv420p",
                                "-preset", "slow", "-crf", "32",
                                video_path
                            ], stdin=subprocess.PIPE)

                        ffmpeg.stdin.write(frame.tobytes())
                if done:
                    print("    bodies sample:", [self.env.robot.bodies[i] for i in range(len(self.env.robot.bodies))], 
                        " episode reward:", f"{episode_reward:.2f}", 
                        " custom reward:", f"{custom_reward:.2f}",
                        " episode length:", t+1,
                        " execute reward:", f"{episode_reward-custom_reward:.2f}")
                    break

                state = next_state

        if ffmpeg is not None:
            ffmpeg.stdin.close()
            ffmpeg.wait()
            print(f"[OK] Video saved to: {video_path}")
            
