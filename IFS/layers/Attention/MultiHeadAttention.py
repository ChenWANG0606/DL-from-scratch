from torch import nn
import numpy as np
import math
import torch

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, num_heads, dropout = 0.1):
        """
        Construct a new MultiHeadAttention layer.

        Inputs:
         - embed_dim: Dimension of the token embedding
         - num_heads: Number of attention heads
         - dropout: Dropout probability
        """
        super().__init__()
        assert emb_dim%num_heads == 0
        self.emb_dim = emb_dim
        self.n_heads = num_heads
        self.head_dim = emb_dim//num_heads

        self.attn_dropout = nn.Dropout(dropout)

        self.query_proj = nn.Linear(emb_dim, emb_dim)
        self.key_proj = nn.Linear(emb_dim, emb_dim)
        self.value_proj = nn.Linear(emb_dim, emb_dim)
        self.proj = nn.Linear(emb_dim, emb_dim)
        

    
    def forward(self, query, key, value, attn_mask = None):
        """
        Calculate the masked attention output for the provided data, computing
        all attention heads in parallel.

        In the shape definitions below, N is the batch size, S is the source
        sequence length, T is the target sequence length, and E is the embedding
        dimension.

        Inputs:
        - query: Input data to be used as the query, of shape (N, S, E)
        - key: Input data to be used as the key, of shape (N, T, E)
        - value: Input data to be used as the value, of shape (N, T, E)
        - attn_mask: Array of shape (S, T) where mask[i,j] == 0 indicates token
          i in the source should not influence token j in the target.

        Returns:
        - output: Tensor of shape (N, S, E) giving the weighted combination of
          data in value according to the attention weights calculated using key
          and query.
        """
        N, S, E = query.shape
        N, T, E = key.shape

        q = self.query_proj(query)
        k = self.key_proj(key)
        v = self.value_proj(value)

        q = q.view(N, S, self.n_heads, self.head_dim).transpose(1,2)
        v = v.view(N, T, self.n_heads, self.head_dim).transpose(1,2)
        k = k.view(N, T, self.n_heads, self.head_dim).transpose(1,2)

        score = q @ k.transpose(-1, -2) / math.sqrt(self.head_dim)

        if attn_mask:
            score = score.masked_fill(score == 0, float('-inf'))
        
        attn_weights = torch.softmax(score, dim = -1)
        attn_weights = self.attn_dropout(attn_weights)

        Y = attn_weights @ v
        Y = Y.transpose(1, 2).contiguous().view(N, S, E)
        output = self.proj(Y)

        return output

if __name__ == "__main__":
    # ===== 测试 =====
    torch.manual_seed(0)

    embed_dim = 32
    num_heads = 4
    batch = 2
    seq_len = 5

    x = torch.randn(batch, seq_len, embed_dim)

    # 自己的
    my_attn = MultiHeadAttention(embed_dim, num_heads)

    # 官方
    torch_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, bias=False)

    # ===== 权重对齐（关键！否则没法比）=====
    with torch.no_grad():
        torch_attn.in_proj_weight.copy_(
            torch.cat([
                my_attn.q_proj.weight,
                my_attn.k_proj.weight,
                my_attn.v_proj.weight
            ], dim=0)
        )
        torch_attn.out_proj.weight.copy_(my_attn.out_proj.weight)

    # ===== forward =====
    out_my = my_attn(x)
    out_torch, _ = torch_attn(x, x, x)

    # ===== 对比 =====
    print("difference:", (out_my - out_torch).abs().max())from torch import nn
import numpy as np
import math
import torch

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, num_heads, dropout = 0.1):
        """
        Construct a new MultiHeadAttention layer.

        Inputs:
         - embed_dim: Dimension of the token embedding
         - num_heads: Number of attention heads
         - dropout: Dropout probability
        """
        super().__init__()
        assert emb_dim%num_heads == 0
        self.emb_dim = emb_dim
        self.n_heads = num_heads
        self.head_dim = emb_dim//num_heads

        self.attn_dropout = nn.Dropout(dropout)

        self.query_proj = nn.Linear(emb_dim, emb_dim)
        self.key_proj = nn.Linear(emb_dim, emb_dim)
        self.value_proj = nn.Linear(emb_dim, emb_dim)
        self.proj = nn.Linear(emb_dim, emb_dim)
        

    
    def forward(self, query, key, value, attn_mask = None):
        """
        Calculate the masked attention output for the provided data, computing
        all attention heads in parallel.

        In the shape definitions below, N is the batch size, S is the source
        sequence length, T is the target sequence length, and E is the embedding
        dimension.

        Inputs:
        - query: Input data to be used as the query, of shape (N, S, E)
        - key: Input data to be used as the key, of shape (N, T, E)
        - value: Input data to be used as the value, of shape (N, T, E)
        - attn_mask: Array of shape (S, T) where mask[i,j] == 0 indicates token
          i in the source should not influence token j in the target.

        Returns:
        - output: Tensor of shape (N, S, E) giving the weighted combination of
          data in value according to the attention weights calculated using key
          and query.
        """
        N, S, E = query.shape
        N, T, E = key.shape

        q = self.query_proj(query)
        k = self.key_proj(key)
        v = self.value_proj(value)

        q = q.view(N, S, self.n_heads, self.head_dim).transpose(1,2)
        v = v.view(N, T, self.n_heads, self.head_dim).transpose(1,2)
        k = k.view(N, T, self.n_heads, self.head_dim).transpose(1,2)

        score = q @ k.transpose(-1, -2) / math.sqrt(self.head_dim)

        if attn_mask:
            score = score.masked_fill(score == 0, float('-inf'))
        
        attn_weights = torch.softmax(score, dim = -1)
        # 注意先经过softmax后dropout，因为drop是吧一部分变成0，不然就没有意义了
        attn_weights = self.attn_dropout(attn_weights)

        Y = attn_weights @ v
        Y = Y.transpose(1, 2).contiguous().view(N, S, E)
        output = self.proj(Y)

        return output

if __name__ == "main":
    # ===== 测试 =====
    torch.manual_seed(0)

    embed_dim = 32
    num_heads = 4
    batch = 2
    seq_len = 5

    x = torch.randn(batch, seq_len, embed_dim)

    # 自己的
    my_attn = MultiHeadAttention(embed_dim, num_heads)

    # 官方
    torch_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, bias=False)

    # ===== 权重对齐（关键！否则没法比）=====
    with torch.no_grad():
        torch_attn.in_proj_weight.copy_(
            torch.cat([
                my_attn.q_proj.weight,
                my_attn.k_proj.weight,
                my_attn.v_proj.weight
            ], dim=0)
        )
        torch_attn.out_proj.weight.copy_(my_attn.out_proj.weight)

    # ===== forward =====
    out_my = my_attn(x)
    out_torch, _ = torch_attn(x, x, x)

    # ===== 对比 =====
    print("difference:", (out_my - out_torch).abs().max())