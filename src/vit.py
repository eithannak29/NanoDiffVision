import pytorch_lightning as pl
import torch
from torch import nn
from layers import PatchEmbeddings, MLP
from attention import MultiHeadAttention, MultiHeadDiffAttention


class TransformerEncoder(nn.Module):
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
        self.attention = (
            MultiHeadDiffAttention(dim, num_heads, layer_idx)
            if use_diff_attention
            else MultiHeadAttention(dim, num_heads)
        )
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

        self.patch_embeddings = PatchEmbeddings(in_channels, patch_size, embedding_dim)
        num_patches = (image_size // patch_size) ** 2
        self.positional_embeddings = nn.Parameter(
            torch.randn(1, 1 + num_patches, embedding_dim)
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))

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

        self.mlp_head = MLP(embedding_dim, hidden_dim, out_dim)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embeddings(x)
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
        x = x + self.positional_embeddings
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.transformer_encoder(x)
        x = x[:, 0]
        x = self.mlp_head(x)
        return x

    def _shared_step(self, batch, prefix):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss(label_smoothing=0.1)(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
        acc = (preds == y).float().mean()
        self.log(f"{prefix}_loss", loss, prog_bar=True, on_epoch=True)
        self.log(f"{prefix}_accuracy", acc, prog_bar=True, on_epoch=True)
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, _ = self._shared_step(batch, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=3e-4, weight_decay=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return [optimizer], [scheduler]
