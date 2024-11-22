import torch
from torch import nn


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


class MultiHeadDiffAttention(nn.Module):
    def __init__(self, dim, num_heads, layer_idx, num_groups=4):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.head_dim = dim // num_heads

        self.K1 = nn.Linear(in_features=dim, out_features=dim, bias=True)
        self.Q1 = nn.Linear(in_features=dim, out_features=dim, bias=False)

        self.K2 = nn.Linear(in_features=dim, out_features=dim, bias=True)
        self.Q2 = nn.Linear(in_features=dim, out_features=dim, bias=False)

        self.V = nn.Linear(in_features=dim, out_features=dim, bias=False)

        self.lambda_q1 = nn.Parameter(torch.randn(num_heads, 1))
        self.lambda_k1 = nn.Parameter(torch.randn(num_heads, 1))
        self.lambda_q2 = nn.Parameter(torch.randn(num_heads, 1))
        self.lambda_k2 = nn.Parameter(torch.randn(num_heads, 1))

        self.lambda_init = 0.8 - 0.6 * torch.exp(torch.tensor(-0.3 * (layer_idx - 1)))

        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=dim)

        self.out_proj = nn.Linear(in_features=dim, out_features=dim)

    def compute_lambda(self):
        lambda_1 = (
            torch.exp(self.lambda_q1 * self.lambda_k1)
            - torch.exp(self.lambda_q2 * self.lambda_k2)
            + self.lambda_init
        )
        return lambda_1.view(1, self.num_heads, 1, 1)

    def _reshape_and_transpose(self, tensor, batch_size, seq_len):
        return tensor.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

    def forward(self, x):
        K1 = self.K1(x)
        Q1 = self.Q1(x)
        K2 = self.K2(x)
        Q2 = self.Q2(x)
        V = self.V(x)

        batch_size, seq_len, _ = x.shape
        K1 = self._reshape_and_transpose(K1, batch_size, seq_len)
        Q1 = self._reshape_and_transpose(Q1, batch_size, seq_len)
        K2 = self._reshape_and_transpose(K2, batch_size, seq_len)
        Q2 = self._reshape_and_transpose(Q2, batch_size, seq_len)
        V = self._reshape_and_transpose(V, batch_size, seq_len)

        s = self.head_dim**0.5

        attention_1 = torch.softmax(torch.matmul(Q1, K1.transpose(-2, -1)) / s, dim=-1)
        attention_2 = torch.softmax(torch.matmul(Q2, K2.transpose(-2, -1)) / s, dim=-1)

        lambda_1 = self.compute_lambda()

        attention = attention_1 - lambda_1 * attention_2

        x = torch.matmul(attention, V)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.dim)

        x = x.transpose(1, 2)
        x = self.norm(x)
        x = x.transpose(1, 2)
        x = self.out_proj(x)

        return x
