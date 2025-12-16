import torch
import torch.nn as nn
import torch.nn.functional as F

class SimilarityNet(nn.Module):
    def __init__(self, embed_dim=768, dropout=0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, embed_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(embed_dim // 2),
            nn.Linear(embed_dim // 2, embed_dim)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, image_embeds, text_embeds):
        v_proj = self.proj(image_embeds)
        t_proj = self.proj(text_embeds)

        v_proj = F.normalize(v_proj, p=2, dim=-1)
        t_proj = F.normalize(t_proj, p=2, dim=-1)
        return v_proj, t_proj
