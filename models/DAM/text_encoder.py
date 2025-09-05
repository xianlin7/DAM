import torch
import torch.nn as nn
from collections import OrderedDict


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask=None):
        super(ResidualAttentionBlock, self).__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.n_head = n_head

    def attention(self, x: torch.Tensor, attn_mask_: torch.Tensor):
        attn_mask_ = attn_mask_.repeat_interleave(self.n_head, dim=0)
        attn_mask_ = attn_mask_.to(dtype=x.dtype, device=x.device) if attn_mask_ is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask_)[0]

    def forward(self, para_tuple: tuple):
        x, attn_mask = para_tuple #[L b D], [b L L]
        x = x + self.attention(self.ln_1(x), attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return (x, attn_mask)


class TextTransformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int):
        super(TextTransformer, self).__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads) for _ in range(layers)])

    def forward(self, x, attn_mask):
        for i in range(self.layers):
            x, attn_mask =  self.resblocks[i]((x, attn_mask))
        return x


class TextEncoder(nn.Module):
    def __init__(self, in_channels=768, out_channels=512, text_length=20, text_layers=4, text_heads=8, pre_pro=True): 
        super().__init__()

        self.pre_pro = pre_pro
        self.embed_dim = out_channels if self.pre_pro else in_channels
        self.positional_embedding = nn.Parameter(torch.randn(1, text_length, self.embed_dim))
        self.transformer = TextTransformer(width=self.embed_dim, layers=text_layers, heads=text_heads)
        self.project = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1) if out_channels != in_channels else nn.Identity()
       
    def build_attention_mask(self, context_length):
        mask = torch.zeros(context_length, context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  
        return mask
    
    def forward(self, x, mask):

        attn_mask = self.build_attention_mask(x.size(1)).repeat(x.size(0), 1, 1).to(mask.device) # [b L L]
        inf = torch.zeros((x.size(1), x.size(1))).fill_(float("-inf")).repeat(x.size(0), 1, 1).to(mask.device)
        mask = mask.unsqueeze(1).expand(-1, mask.size(1), -1) # [b L L]
        attn_mask = torch.where(mask>0, attn_mask, inf) # [b L L]

        if self.pre_pro:
            x = self.project(x.transpose(1, 2)).transpose(1, 2) 
        
        pos_emd = self.positional_embedding[:, :x.size(1), :] # [1 L d]
        x = x + pos_emd # [b L d]
        x = x.permute(1, 0, 2)  # [L b d]
        x = self.transformer(x, attn_mask) # [L b d]
        x = x.permute(1, 0, 2)  # [b L d]

        if not self.pre_pro:
            x = self.project(x.transpose(1, 2)).transpose(1, 2) 

        return x


class TextEncoderSimple(nn.Module):
    def __init__(self, in_channels=768, text_dim=512): 
        super().__init__()

        self.text_module4 = nn.Conv1d(in_channels=in_channels, out_channels=in_channels//2, kernel_size=3, padding=1)
        self.text_module3 = nn.Conv1d(in_channels=in_channels//2, out_channels=text_dim, kernel_size=3, padding=1)
        self.text_module2 = nn.Conv1d(in_channels=text_dim, out_channels=text_dim//2, kernel_size=3, padding=1)
        self.text_module1 = nn.Conv1d(in_channels=text_dim//2, out_channels=text_dim, kernel_size=3, padding=1)
    
    def forward(self, text):
        text = self.text_module4(text.transpose(1, 2))
        text = self.text_module3(text) 
        text = self.text_module2(text)
        text = self.text_module1(text).transpose(1, 2) 
        return text
