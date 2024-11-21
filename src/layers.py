from torch import nn


class PatchEmbeddings(nn.Module):
    def __init__(self, in_channels=1, patch_size=8, embedding_dim=128):
        super().__init__()
        self.unfolding = nn.Unfold(kernel_size=patch_size, stride=patch_size)
        self.projection = nn.Linear(in_channels * patch_size**2, embedding_dim)

    def forward(self, x):
        x = self.unfolding(x)  # H * W * C -> N * ( P * P * C)
        x = x.transpose(1, 2)  # N * ( P * P * C) -> N * ( P * P * C)
        x = self.projection(x)  # N * ( P * P * C) -> N * E
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
        x = self.dropout(x)
        return x
