from torch import nn


class PatchEmbeddings(nn.Module):
    def __init__(self, in_channels=1, patch_size=8, embedding_dim=128):
        super().__init__()
        self.projection = nn.Conv2d(in_channels, embedding_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.projection(x)
        B, C, H, W = x.shape
        x = x.view(B,C,H*W).transpose(1,2)
        return x


class MLP(nn.Module):
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
        x = self.activation(x)
        x = self.dropout(x)
        return x
