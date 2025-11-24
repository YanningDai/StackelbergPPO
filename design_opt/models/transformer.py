import torch
import torch.nn.functional as F
from torch import nn
import math
import logging

class MaskedSelfAttention(nn.Module):
    def __init__(self, hidden_dim) -> None:
        super().__init__()
        self.fc_q = nn.Linear(hidden_dim, hidden_dim)
        self.fc_k = nn.Linear(hidden_dim, hidden_dim)
        self.fc_v = nn.Linear(hidden_dim, hidden_dim)
        
    def scaled_dot_product_attention(self, query, key, value, attn_mask=None, dropout_p=0.0):
        L, S = query.size(-2), key.size(-2)
        B = query.size(0)
        scale_factor = 1 / math.sqrt(query.size(-1))
        attn_bias = torch.zeros(B, L, S, dtype=query.dtype, device=query.device)
        if attn_mask is not None and attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            raise NotImplementedError
        attn_weight = query @ key.transpose(-2, -1) * scale_factor
        attn_weight = attn_weight + attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
        return attn_weight @ value
    
    def forward(self, x, attn_mask=None):
        q, k, v = self.fc_q(x), self.fc_k(x), self.fc_v(x)
        
        out = self.scaled_dot_product_attention(q, k, v, attn_mask)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, norm_type="pre", act_layer=nn.SiLU) -> None:
        super().__init__()
        HIDDEN_RATIO = 4
        self.norm_type = norm_type
        
        self.attn = MaskedSelfAttention(hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * HIDDEN_RATIO),
            act_layer(),
            nn.Linear(hidden_dim * HIDDEN_RATIO, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
    
    def forward(self, x, attn_mask=None):
        if self.norm_type == "post":
            x = self.norm1(x + self.attn(x, attn_mask))
            x = self.norm2(x + self.mlp(x))
        elif self.norm_type == "pre":
            x = x + self.attn(self.norm1(x), attn_mask)
            x = x + self.mlp(self.norm2(x))
        else:
            raise NotImplementedError
        
        return x
    
class TransformerSimple(nn.Module):
    def __init__(self, in_dim, cfg, node_dim=0, lapPE_k=4, embedding_len=256) -> None:
        super(TransformerSimple, self).__init__()
        self.cfg = cfg
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim = cfg['hidden_dim']
        self.num_layers = num_layers = cfg['block_depth']
        
        self.norm_type = norm_type = cfg['norm_type']
        self.pos_emb_type = pos_emb_type = cfg['pos_emb_type']
        self.embedding_len = embedding_len
        self.out_dim = hidden_dim
        
        self.in_fc = nn.Linear(in_dim, hidden_dim)
        
        if pos_emb_type == "index":
            self.index_embedding = nn.Embedding(embedding_len, hidden_dim)
        elif pos_emb_type == "travel":
            self.travel_embedding = nn.Embedding(embedding_len, hidden_dim)
            
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, norm_type) for _ in range(num_layers)
        ])
        
    def forward(self, transformer_obs):
        x = transformer_obs["padded_obs"]
        padding_mask = transformer_obs["padding_mask"]  
        padded_body_ind = transformer_obs["padded_body_ind"]  
        
        # project to hidden dimension
        x = self.in_fc(x)

        # position embedding
        if self.pos_emb_type == "index":
            pos_emb = self.index_embedding(padded_body_ind)
            x = x + pos_emb
        elif self.pos_emb_type == "travel":
            B, L, D = x.shape
            position_indices = torch.arange(0, L, dtype=torch.long, device=x.device).unsqueeze(0)
            pos_emb = self.travel_embedding(position_indices)
            x = x + pos_emb
        
        ## attention mask for padding
        attn_mask = padding_mask.unsqueeze(1) if padding_mask is not None else None
        
        for block in self.blocks:
            x = block(x, attn_mask=attn_mask)
            
        x = x[padding_mask]
        
        return x