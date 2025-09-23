import torch
from torch import nn

torch.manual_seed(42)

class Embedded_Patches(nn.Module):
    def __init__(self, in_ch, patch_size, emb_size, img_size):
        super().__init__()
        num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(in_ch, emb_size, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.pos_embed = nn.Parameter(torch.randn((1, 1+num_patches, emb_size)))

    def forward(self, x):
        B = x.shape[0]
        x = self.proj(x)        # (B, emb_size, H/patch_size, W/patch_size)
        x = x.flatten(2)        # (B, emb_size, num_patches)
        x = x.transpose(1, 2)   # (B, num_patches, emb_size)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, emb_size)
        x = torch.cat((cls_tokens, x), dim=1)          # (B, 1+num_patches, emb_size)
        x = x + self.pos_embed                         # (B, 1+num_patches, emb_size)
        return x

class MLP(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features,
                 dropout=0.):
        super().__init__()

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.gelu = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.drop(self.gelu(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x

class Transofrmer_Encoder(nn.Module):
    def __init__(self, emb_size, num_heads, mlp_dim, dropout=0.):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(emb_size)
        self.attn = nn.MultiheadAttention(emb_size, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(emb_size)
        self.mlp = MLP(emb_size, mlp_dim, dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

class ViT(nn.Module):
    def __init__(self, 
                 in_ch,
                 patch_size,
                 emb_size,
                 img_size,
                 num_heads,
                 mlp_dim,
                 depth,
                 num_classes,
                 dropout=0.):
        super().__init__()
        
        self.embedded_patches = Embedded_Patches(in_ch, patch_size, emb_size, img_size)
        self.encoder_layers = nn.ModuleList([
            Transofrmer_Encoder(emb_size=emb_size, num_heads=num_heads, mlp_dim=mlp_dim, dropout=dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(emb_size)
        self.head = nn.Linear(emb_size, num_classes)

    def forward(self, x):
        x = self.embedded_patches(x)    # (B, 1+num_patches, emb_size)
        for layer in self.encoder_layers:
            x = layer(x)

        x = self.norm(x)
        cls_token_final = x[:, 0]       # (B, emb_size)
        x = self.head(cls_token_final)  # (B, num_classes)
        return x

