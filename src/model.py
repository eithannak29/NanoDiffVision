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
    def __init__(self, dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        assert dim % n_heads == 0
        self.head_dim = dim // n_heads

        self.K = nn.Linear(in_features=dim, out_features=dim)
        self.Q = nn.Linear(in_features=dim, out_features=dim)
        self.V = nn.Linear(in_features=dim, out_features=dim)

        self.out_proj = nn.Linear(in_features=dim, out_features=dim)

    def forward(self, x):
        K = self.K(x)
        Q = self.Q(x)
        V = self.V(x)

        K = K.view(x.shape[0], x.shape[1], self.n_heads, self.head_dim).transpose(1, 2)
        Q = Q.view(x.shape[0], x.shape[1], self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(x.shape[0], x.shape[1], self.n_heads, self.head_dim).transpose(1, 2)

        attention = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim**0.5)
        attention = torch.nn.functional.softmax(attention, dim=-1)
        x = torch.matmul(attention, V)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.dim)
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
    def __init__(self, dim, hidden_dim, n_heads, dropout=0.1):
        super().__init__()
        self.ln_pre_attn = nn.LayerNorm(dim)
        self.attention = MultiHeadAttention(dim, n_heads)
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
        shape=32,
        patch_size=4,
        embedding_dim=384,
        hidden_dim=1024,
        n_blocks=7,
        n_heads=6,
        out_dim=10,
        dropout=0.1,
    ):
        super().__init__()

        # Patch Embeddings
        self.patch_embeddings = PatchEmbeddings(
            in_channels=in_channels, patch_size=patch_size, embedding_dim=embedding_dim
        )

        # Positional Embeddings
        num_patches = (shape // patch_size) ** 2
        self.positional_embeddings = nn.Parameter(
            torch.randn(1, 1 + num_patches, embedding_dim)
        )

        # Class Token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))

        # Transformer Encoder
        self.transformer_encoder = nn.Sequential(
            *[
                TransformerEncoder(embedding_dim, hidden_dim, n_heads, dropout)
                for _ in range(n_blocks)
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
