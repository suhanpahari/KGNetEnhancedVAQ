"""Multi-modal fusion layer with cross-attention."""

import torch
import torch.nn as nn


class MultiModalFusionLayer(nn.Module):
    """Fuses vision, text, and knowledge features."""

    def __init__(self, vision_dim=768, text_dim=768, kg_dim=768, hidden_dim=1024, num_heads=8):
        super().__init__()
        
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.kg_proj = nn.Linear(kg_dim, hidden_dim)
        
        self.vision_text_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.vision_kg_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.text_kg_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 3),
            nn.Softmax(dim=-1)
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, vision_feats, text_feats, kg_feats):
        """
        Args:
            vision_feats: (B, vision_dim)
            text_feats: (B, text_dim)
            kg_feats: (B, kg_dim)
        Returns:
            fused_features: (B, hidden_dim)
        """
        # Project to common space
        v = self.vision_proj(vision_feats).unsqueeze(1)  # (B, 1, hidden_dim)
        t = self.text_proj(text_feats).unsqueeze(1)
        k = self.kg_proj(kg_feats).unsqueeze(1)
        
        # Cross-attention
        vt_attn, _ = self.vision_text_attn(v, t, t)
        vk_attn, _ = self.vision_kg_attn(v, k, k)
        tk_attn, _ = self.text_kg_attn(t, k, k)
        
        vt_attn = vt_attn.squeeze(1)
        vk_attn = vk_attn.squeeze(1)
        tk_attn = tk_attn.squeeze(1)
        
        # Adaptive fusion
        concat_feats = torch.cat([vt_attn, vk_attn, tk_attn], dim=-1)
        weights = self.fusion_gate(concat_feats)  # (B, 3)
        
        fused = (weights[:, 0:1] * vt_attn + 
                weights[:, 1:2] * vk_attn + 
                weights[:, 2:3] * tk_attn)
        
        fused = self.layer_norm(fused)
        
        return fused
