import numpy as np
import tempfile
import torch
import os
import random
from datasketch import MinHash, MinHashLSH
import math
class TrajBatchDisc:

    def __init__(self, memory_list):
        memory = memory_list[0]
        for x in memory_list[1:]:
            memory.append(x)
        self.batch = zip(*memory.sample())
        self.states = list(next(self.batch))
        self.actions = list(next(self.batch))
        self.next_terminations = np.stack(next(self.batch))
        self.next_dones = np.stack(next(self.batch))
        self.next_states = list(next(self.batch))
        self.rewards = np.stack(next(self.batch))
        self.exps = np.stack(next(self.batch))
        self.c_reward = np.stack(next(self.batch))

def init_fc_weights(fc):
    fc.weight.data.mul_(0.1) 
    fc.bias.data.mul_(0.0)
    

def mask_list(lst, mask):
    return [item for i, item in enumerate(lst) if mask[i]]

def set_global_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

def deterministic_cumsum(tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
    if torch.are_deterministic_algorithms_enabled():
        x_cpu = tensor.cpu()
        y_cpu = torch.cumsum(x_cpu, dim=dim)
        return y_cpu.to(tensor.device)
    else:
        return torch.cumsum(tensor, dim=dim)

class CheckpointManager:
    @staticmethod
    def save_temp(model, prefix='checkpoint'):
        fd, path = tempfile.mkstemp(prefix=f'{prefix}_', suffix='.pth')
        torch.save(model.state_dict(), path)
        os.close(fd)
        return path

    @staticmethod
    def load_temp(model, path):
        state_dict = torch.load(path, weights_only=True)
        model.load_state_dict(state_dict)
        return model

    @staticmethod
    def cleanup(paths):
        
        if isinstance(paths, dict):
            paths = list(paths.values())
        
        if isinstance(paths, (list, tuple)):
            for path in paths:
                if isinstance(path, str) and os.path.exists(path):
                    os.remove(path)
        elif isinstance(paths, str) and os.path.exists(paths):
            os.remove(paths)
            
            

    def _pad_to_max_node(self, lapPE):        
        assert self.pe_dim, "pe_dim is not set"

        padded_list = []

        for graph_pe in lapPE:
            assert graph_pe.shape[0] <= self.max_node, "graph_pe exceeds max_node limit"
            graph_pe_cpu = graph_pe.cpu() 
            padded = torch.zeros((self.max_node, self.pe_dim),
                                 device='cpu', 
                                 dtype=graph_pe_cpu.dtype)
            padded[:graph_pe_cpu.shape[0]] = graph_pe_cpu
            padded_list.append(padded.flatten())  # append the flattened 1D tensor to the list

        return torch.stack(padded_list)

    def _vector_to_minhash(self, vector):
        # vector: 1D Tensor, e.g., 60 dims
        tokens = [str(int(x.item() * 100)) for x in vector]
        m = MinHash(num_perm=self.num_perm)
        for token in tokens:
            m.update(token.encode('utf8'))
        return m
    

class EpisodeBatchPlanner:
    """
      1) Extract per-episode indices for leaders and followers from `episodes` + `state_types`.
      2) Pre-generate a paired follower matrix `Fmat` with shape [E, m*n] for an entire epoch (pad if needed).
      3) Aggregate leaders into minibatches by episode (do not cut episodes) and provide paired follower indices for each round k.
    """

    def __init__(self, m, n, pad_value=-1, shuffle_episode=True):
        self.m = int(m)                  # number of follower steps taken per episode each round for Stackelberg
        self.n = int(n)                  # number of optimization rounds per epoch (opt_num_epochs)
        self.pad_value = int(pad_value)  # padding value for Fmat
        self.shuffle_episode = bool(shuffle_episode)

    @staticmethod
    def build_episode_indices(episodes, state_types_np):
        """Return two lists; each element is a numpy array of global indices for that episode."""
        leader_idx_ep = [ep[state_types_np[ep] != 2] for ep in episodes]
        follower_idx_ep = [ep[state_types_np[ep] == 2] for ep in episodes]
        return leader_idx_ep, follower_idx_ep

    def make_Fmat(self, follower_idx_ep):
        """Fmat: [E, m*n], where each column block corresponds to the follower sampling window used in one round (k)."""
        E = len(follower_idx_ep)
        K = self.m * self.n
        Fmat = np.full((E, K), self.pad_value, dtype=np.int64)
        for e in range(E):
            idx = np.asarray(follower_idx_ep[e])
            assert idx.size > 0, f"[Fmat] episode {e} has no follower steps; not filtered earlier."
            if idx.size >= K:
                Fmat[e, :] = np.random.permutation(idx)[:K]
            else:
                Fmat[e, :] = np.random.choice(idx, size=K, replace=True)
        return Fmat

    @staticmethod
    def flatten_nonempty(list_of_arrays):
        arrs = [x for x in list_of_arrays if x is not None and len(x) > 0]
        return np.concatenate(arrs, axis=0) if arrs else np.asarray([], dtype=np.int64)

    def iter_leader_batches(self, leader_idx_ep, Fmat, k, MB):
        """
        Get (leader_idx_np, follower_idx_np) minibatches:
          - Leaders are concatenated by episode, aiming to approach MB without splitting episodes.
          - Followers come from the k-th window slice of Fmat (m per episode), with padding removed.
        """
        E = len(leader_idx_ep)
        ep_order = np.random.permutation(E) if self.shuffle_episode else np.arange(E)

        # Convert the k-th round follower slice into per-episode 1D arrays
        fol_slice = Fmat[:, k*self.m:(k+1)*self.m]
        fol_slice_ep = [row[row != self.pad_value] for row in fol_slice]

        cur_L, cur_F = [], []
        cur_cnt = 0

        for e in ep_order:
            L_ep = np.asarray(leader_idx_ep[e])
            if L_ep.size == 0:
                continue
            F_ep = np.asarray(fol_slice_ep[e])  
            add = L_ep.size

            if cur_cnt > 0 and cur_cnt + add > MB:
                yield (np.concatenate(cur_L, axis=0),
                       np.concatenate(cur_F, axis=0) if len(cur_F) > 0 else np.asarray([], dtype=np.int64))
                cur_L, cur_F, cur_cnt = [], [], 0

            cur_L.append(L_ep)
            if F_ep.size > 0:
                cur_F.append(F_ep)
            cur_cnt += add

        if cur_cnt > 0:
            yield (np.concatenate(cur_L, axis=0),
                   np.concatenate(cur_F, axis=0) if len(cur_F) > 0 else np.asarray([], dtype=np.int64))