import torch
from torch import nn

# Classic Attention

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.head_dim = dim // num_heads

        self.K = nn.Linear(in_features=dim, out_features=dim, bias=True)
        self.Q = nn.Linear(in_features=dim, out_features=dim, bias=False)
        self.V = nn.Linear(in_features=dim, out_features=dim, bias=False)

        self.out_proj = nn.Linear(in_features=dim, out_features=dim)

    def _reshape_and_transpose(self, tensor, batch_size, seq_len):
        return tensor.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        K = self._reshape_and_transpose(self.K(x), batch_size, seq_len)
        Q = self._reshape_and_transpose(self.Q(x), batch_size, seq_len)
        V = self._reshape_and_transpose(self.V(x), batch_size, seq_len)

        s = self.head_dim**0.5

        attention = torch.softmax(torch.matmul(Q, K.transpose(-2, -1)) / s, dim=-1)
        x = torch.matmul(attention, V)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.dim)
        x = self.out_proj(x)

        return x
    
    

# Differential Attention

def lambda_init(layer_idx):
    return 0.8 - 0.6 * torch.exp(torch.tensor(-0.3 * (layer_idx - 1)))

def DiffAttention(Q, K, V, lamb, scaling):
    Q1, Q2 = torch.chunk(Q, 2, dim=-1)
    K1, K2 = torch.chunk(K, 2, dim=-1)
    A1 = torch.matmul(Q1, K1.transpose(-1, -2)) * scaling
    A2 = torch.matmul(Q2, K2.transpose(-1, -2)) * scaling
    attention = torch.softmax(A1, dim=-1) - lamb * torch.softmax(A2, dim=-1)
    output = torch.matmul(attention, V)
    return output

class MultiHeadDiffAttention(nn.Module):
    def __init__(self, dim, num_heads, layer_idx):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.head_dim = dim // num_heads

        self.q_proj = nn.Linear(dim, dim * 2, bias=False)
        self.k_proj = nn.Linear(dim, dim * 2, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        self.scaling = self.head_dim**-0.5

        self.lambda_init = lambda_init(layer_idx)

        self.lambda_q1 = nn.Parameter(
            torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1)
        )
        self.lambda_k1 = nn.Parameter(
            torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1)
        )
        self.lambda_q2 = nn.Parameter(
            torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1)
        )
        self.lambda_k2 = nn.Parameter(
            torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1)
        )

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, 2 * self.head_dim)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, 2 * self.head_dim)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1))
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1))
        lamb = lambda_1 - lambda_2 + self.lambda_init

        attn_output = DiffAttention(Q, K, V, lamb, self.scaling)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)

        output = self.out_proj(attn_output)
        output = self.norm(output)
        
        return output
