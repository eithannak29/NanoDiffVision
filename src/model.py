import pytorch_lightning as pl
import torch
from torch import nn


class PatchEmbeddings(pl.LightningModule):
    def __init__(self, in_channels=1, patch_size=8, embedding_dim=128):
        super().__init__()
        self.unfolding = nn.Unfold(kernel_size=patch_size, stride=patch_size)
        self.projection = nn.Linear(in_channels * patch_size**2, embedding_dim)

    def forward(self, x):
        x = self.unfolding(x)  # H * W * C -> N * ( P * P * C)
        x = x.transpose(1, 2)  # N * ( P * P * C) -> N * ( P * P * C)
        x = self.projection(x)  # N * ( P * P * C) -> N * E
        return x


class MultiHeadAttention(pl.LightningModule):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        assert dim % num_heads == 0
        self.head_dim = dim // num_heads

        self.K = nn.Linear(in_features=dim, out_features=dim)
        self.Q = nn.Linear(in_features=dim, out_features=dim)
        self.V = nn.Linear(in_features=dim, out_features=dim)

        self.out_proj = nn.Linear(in_features=dim, out_features=dim)

    def _reshape_and_transpose(self, tensor, batch_size, seq_len):
        return tensor.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(
            1, 2
        )

    def forward(self, x):
        K = self._reshape_and_transpose(self.K(x))
        Q = self._reshape_and_transpose(self.Q(x))
        V = self._reshape_and_transpose(self.V(x))

        s = self.head_dim**0.5

        attention = torch.softmax(torch.matmul(Q, K.transpose(-2, -1)) / s, dim=-1)
        x = torch.matmul(attention, V)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.dim)
        x = self.out_proj(x)

        return x


class MultiHeadDiffAttention(pl.LightningModule):
    def __init__(self, dim, num_heads, layer_idx):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.head_dim = dim // num_heads

        self.K1 = nn.Linear(in_features=dim, out_features=dim)
        self.Q1 = nn.Linear(in_features=dim, out_features=dim)

        self.K2 = nn.Linear(in_features=dim, out_features=dim)
        self.Q2 = nn.Linear(in_features=dim, out_features=dim)

        self.V = nn.Linear(in_features=dim, out_features=dim)

        self.lambda_q1 = nn.Parameter(torch.randn(num_heads, 1))
        self.lambda_k1 = nn.Parameter(torch.randn(num_heads, 1))
        self.lambda_q2 = nn.Parameter(torch.randn(num_heads, 1))
        self.lambda_k2 = nn.Parameter(torch.randn(num_heads, 1))

        self.lambda_init = 0.8 - 0.6 * torch.exp(torch.tensor(-0.3 * (layer_idx - 1)))

        self.norm = nn.LayerNorm(dim)

        self.out_proj = nn.Linear(in_features=dim, out_features=dim)

    def compute_lambda(self):
        lambda_1 = (
            torch.exp(self.lambda_q1 * self.lambda_k1)
            - torch.exp(self.lambda_q2 * self.lambda_k2)
            + self.lambda_init
        )
        return lambda_1.view(1, self.num_heads, 1, 1)

    def _reshape_and_transpose(self, tensor, batch_size, seq_len):
        return tensor.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(
            1, 2
        )

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
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.dim)
        x = self.norm(x)
        x = self.out_proj(x)

        return x


class MLP(pl.LightningModule):
    def __init__(
        self,
        in_features,
        hidden_features,
        out_features,
        dropout=0.1,
        activation=nn.GELU,
    ):
        super().__init__()
        self.fc1 = nn.Linear(in_features=in_features, out_features=hidden_features)
        self.fc2 = nn.Linear(in_features=hidden_features, out_features=out_features)
        self.dropout = nn.Dropout(p=dropout)
        self.activation = activation()

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerEncoder(pl.LightningModule):
    def __init__(
        self,
        dim,
        hidden_dim,
        num_heads,
        dropout=0.1,
        use_diff_attention=False,
        layer_idx=1,
    ):
        super().__init__()
        self.ln_pre_attn = nn.LayerNorm(dim)

        if use_diff_attention:
            self.attention = MultiHeadDiffAttention(dim, num_heads, layer_idx)
        else:
            self.attention = MultiHeadAttention(dim, num_heads)

        self.ln_pre_ffn = nn.LayerNorm(dim)
        self.ffn = MLP(dim, hidden_dim, dim, dropout)

    def forward(self, x):
        x = x + self.attention(self.ln_pre_attn(x))
        x = x + self.ffn(self.ln_pre_ffn(x))
        return x


class ViT(pl.LightningModule):
    def __init__(
        self,
        in_channels=1,
        image_size=32,
        patch_size=4,
        embedding_dim=384,
        hidden_dim=1024,
        num_blocks=7,
        num_heads=6,
        out_dim=10,
        dropout=0.1,
        use_diff_attention=False,
    ):
        super().__init__()

        # Patch Embeddings
        self.patch_embeddings = PatchEmbeddings(
            in_channels=in_channels, patch_size=patch_size, embedding_dim=embedding_dim
        )

        # Positional Embeddings
        num_patches = (image_size // patch_size) ** 2
        self.positional_embeddings = nn.Parameter(
            torch.randn(1, 1 + num_patches, embedding_dim)
        )

        # Class Token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))

        # Transformer Encoder
        self.transformer_encoder = nn.Sequential(
            *[
                TransformerEncoder(
                    embedding_dim,
                    hidden_dim,
                    num_heads,
                    dropout,
                    use_diff_attention,
                    layer_idx=i,
                )
                for i in range(num_blocks)
            ]
        )

        # MLP Head
        self.mlp_head = MLP(embedding_dim, embedding_dim * 4, out_dim)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embeddings(x)
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
        x = x + self.positional_embeddings
        x = self.transformer_encoder(x)
        x = x[:, 0]
        x = self.mlp_head(x)
        return x

    def _shared_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss(label_smoothing=0.1)(y_hat, y)
        return loss, y_hat, y

    def training_step(self, batch, batch_idx):
        loss, y_hat, y = self._shared_step(batch, batch_idx)
        preds = torch.argmax(y_hat, dim=1)
        acc = (preds == y).float().mean()
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        self.log("train_accuracy", acc, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, y_hat, y = self._shared_step(batch, batch_idx)
        preds = torch.argmax(y_hat, dim=1)
        acc = (preds == y).float().mean()
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_accuracy", acc, prog_bar=True, on_epoch=True)
        return {"val_loss": loss, "val_accuracy": acc}

    def test_step(self, batch, batch_idx):
        loss, y_hat, y = self._shared_step(batch, batch_idx)
        preds = torch.argmax(y_hat, dim=1)
        acc = (preds == y).float().mean()
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_accuracy", acc, prog_bar=True)
        return {"test_loss": loss, "test_accuracy": acc}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=3e-4, weight_decay=0.1)
        return optimizer
