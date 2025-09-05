""" Components of the alignment module of DAM """
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torchvision.ops import DeformConv2d
from collections import OrderedDict

# ------------------------------------------------ for object alignment module ------------------------------------------------
def softmax_one(x, dim=None):
    x = x - x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x)
    return exp_x / (1 + exp_x.sum(dim=dim, keepdim=True))

class MLP(nn.Module):

    def __init__(self, in_channels, out_channels, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(in_channels)
        self.net = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(out_channels, out_channels),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.norm(x)
        x = self.net(x)
        return x


class CrossAttention(nn.Module):
    
    def __init__(self, dim_q, dim_kv, dim_out, heads=8, dim_heads=32, dropout=0.):
        super().__init__()
        inner_dim = dim_heads * heads
        project_out = not (heads == 1 and dim_heads == dim_q)

        self.heads = heads
        self.scale = dim_heads ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.normq = nn.LayerNorm(dim_q)
        self.normkv = nn.LayerNorm(dim_kv)
        self.to_q = nn.Linear(dim_q, inner_dim, bias = False)
        self.to_k = nn.Linear(dim_kv, inner_dim, bias = False)
        self.to_v = nn.Linear(dim_kv, inner_dim, bias = False)
        self.recover_q = nn.Linear(inner_dim, dim_q, bias = False)

        self.to_out = MLP(dim_q, dim_out, dropout=dropout) if project_out else nn.Identity()

    def forward(self, q0, kv0):
        q = self.to_q(self.normq(q0))
        q = rearrange(q, 'b n (g c) -> b g n c', g=self.heads)
        kv = self.normkv(kv0)
        k = self.to_k(kv)
        k = rearrange(k, 'b n (g c) -> b g n c', g=self.heads)
        v = self.to_v(kv)
        v = rearrange(v, 'b n (g c) -> b g n c', g=self.heads)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.recover_q(out) + q0
        out =  out + self.to_out(out)
        return out

#---------------------------------------------------------------------------------------

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)
    

class MaskedCrossAttention(nn.Module):
    
    def __init__(self, dim_q, dim_kv, dim_out, heads=8, dropout=0):
        super(MaskedCrossAttention, self).__init__()

        self.attn = nn.MultiheadAttention(dim_q, heads, kdim=dim_kv, vdim=dim_kv)
        self.ln_q = LayerNorm(dim_q)
        self.ln_kv = LayerNorm(dim_kv)
        # self.mlp = nn.Sequential(OrderedDict([
        #     ("c_fc", nn.Linear(dim_q, dim_q * 4)),
        #     ("gelu", QuickGELU()),
        #     ("c_proj", nn.Linear(dim_q * 4, dim_out))
        # ]))
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(dim_q, dim_q * 4)),
            ("gelu", QuickGELU()),
            ("dropout", nn.Dropout(dropout)),
            ("c_proj", nn.Linear(dim_q * 4, dim_out)),
            ("dropout", nn.Dropout(dropout))
        ]))
        self.ln_2 = LayerNorm(dim_q)
        self.n_head = heads

    def attention(self, query, key, attn_mask_):
        if attn_mask_ is not None:
            attn_mask_ = attn_mask_.repeat_interleave(self.n_head, dim=0)
            attn_mask_ = attn_mask_.to(dtype=query.dtype, device=query.device) if attn_mask_ is not None else None
        # q should be (L, b, E_q), k should be (S, b, E_k), v should be (S, b, E_v), attn_mask should be (b*heads, L, S)
        return self.attn(query, key, key, need_weights=False, attn_mask=attn_mask_)[0]

    def forward(self, query, key, attn_mask):
        """ query:(q b c1), key:(k b c2), attn_mask:(b q k) """
        query = query + self.attention(self.ln_q(query), self.ln_kv(key), attn_mask)
        query = query + self.mlp(self.ln_2(query))
        return (query, key, attn_mask)


class MaskedCrossAttentionBlock(nn.Module):
    def __init__(self, dim_q, dim_kv, dim_out, heads=8, layers=1, dropout=0):
        super(MaskedCrossAttentionBlock, self).__init__()
        self.layers = layers
        self.resblocks = nn.Sequential(*[MaskedCrossAttention(dim_q, dim_kv, dim_out, heads=heads, dropout=dropout) for _ in range(layers)])

    def forward(self, query, key, attn_mask=None):
        for i in range(self.layers):
            query, key, attn_mask =  self.resblocks[i](query, key, attn_mask)
        return query

# ------------------------------------------------ for transformer layer ------------------------------------------------

class PreNorm(nn.Module):

    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


def softmax_one(x, dim=None, _stacklevel=3, dtype=None):
    #subtract the max for stability
    x = x - x.max(dim=dim, keepdim=True).values
    #compute exponentials
    exp_x = torch.exp(x)
    #compute softmax values and add on in the denominator
    return exp_x / (1 + exp_x.sum(dim=dim, keepdim=True))

  
class FeedForward(nn.Module):

    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
  

class Attention(nn.Module):

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        #attn = softmax_one(dots, dim=-1)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):

    def __init__(self, dim, depth, heads, dim_head, mlp_dim=1024, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


#---------------------------------------------------------------------------------------  

class CrossTransformer(nn.Module):
    
    def __init__(self, dim_q, dim_kv, dim_out, heads=8, dim_heads=32, dropout=0.):
        super().__init__()
        inner_dim = dim_heads * heads
        project_out = not (heads == 1 and dim_heads == dim_q)

        self.heads = heads
        self.scale = dim_heads ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.normq1 = nn.LayerNorm(dim_q)
        self.normq2 = nn.LayerNorm(dim_q)
        self.normkv = nn.LayerNorm(dim_kv)
        self.to_q = nn.Linear(dim_q, inner_dim, bias = False)
        self.to_k = nn.Linear(dim_kv, inner_dim, bias = False)
        self.to_v = nn.Linear(dim_kv, inner_dim, bias = False)
        self.recover_q = nn.Linear(inner_dim, dim_q, bias = False)

        self.to_out = MLP(dim_q, dim_out, dropout=dropout) if project_out else nn.Identity()

    def forward(self, q0, kv0):
        q = self.to_q(self.normq1(q0))
        q = rearrange(q, 'b n (g c) -> b g n c', g=self.heads)
        kv = self.normkv(kv0)
        k = self.to_k(kv)
        k = rearrange(k, 'b n (g c) -> b g n c', g=self.heads)
        v = self.to_v(kv)
        v = rearrange(v, 'b n (g c) -> b g n c', g=self.heads)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        #attn = self.attend(dots)
        attn = softmax_one(dots, dim=-1)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.recover_q(out) + q0
        out =  out + self.to_out(self.normq2(out))
        return out


class NoAlighment(nn.Module):

    def __init__(self, dim_image, dim_text, dim_query, dim_alignment, n_token=25, text_ksize=5, num_heads=8, dim_heads=32, window_size=8, window_num=4):
        super().__init__()

    def forward(self, image, text, query):
        return image, text, query


class SimpleAlighment(nn.Module):

    def __init__(self, dim_image, dim_text, dim_query, dim_alignment, n_token=25, text_ksize=5, num_heads=8, dim_heads=32, window_size=8, window_num=4):
        super().__init__()
        self.image_to_object = CrossAttention(dim_q=dim_query, dim_kv=dim_image, dim_out=dim_alignment, heads=num_heads, dim_heads=dim_heads)
        self.text_to_object = CrossAttention(dim_q=dim_query, dim_kv=dim_text, dim_out=dim_alignment, heads=num_heads, dim_heads=dim_heads)
        self.to_query = MLP(2*dim_alignment, dim_query, bias = False)
        self.object_to_text = CrossAttention(dim_q=dim_text, dim_kv=dim_query, dim_out=dim_text, heads=num_heads, dim_heads=dim_heads)
        self.object_to_image = CrossAttention(dim_q=dim_image, dim_kv=dim_query, dim_out=dim_image, heads=num_heads, dim_heads=dim_heads)
       

    def forward(self, image, text, query):
        """ inputs: image:(b c1 H W), text:(b Nt c2), query:(b Nq c3) """
        H, W = image.shape[2], image.shape[3]
        image = rearrange(image, 'b c1 H W -> b (H W) c1')

        query1 = self.object_to_text(query, text)
        query2 = self.image_to_object(query, image)

        query = torch.cat((query1, query2), dim=-1)
        query = self.to_query(query)
        
        text = self.object_to_text(text, query)
        image = self.object_to_image(image, query)
        image = rearrange(image, 'b (H W) c1 -> b c1 H W', H=H, W=W)

        return image, text, query


class WindowRouting(nn.Module):

    def __init__(self, dim, topw=4, diff_routing=False):
        super().__init__()
        self.scale = dim ** -0.5
        self.topw = topw
        self.diff_routing = diff_routing
    
    def forward(self, query, image):
        """
        inputs: query:(b Nq c), image:(b mn c)
        Return: topw_index:(b Nq topw)
        """
        if not self.diff_routing:
            query, image = query.detach(), image.detach()
        attn = torch.matmul(query, image.transpose(-1, -2)) * self.scale # (b Nq mn)
        _, topw_index = torch.topk(attn, k=self.topw, dim=-1) # (b Nq topw)
        return topw_index
    

class ProposalGather(nn.Module):

    def __init__(self, topw):
        super().__init__()
        self.topw = topw

    def forward(self, index, image):
        """
        Inputs: index:(b Nq topw), image:(b mn ws^2 c)
        Return: topw_kv:(b Nq topw ws^2 c)
        """
        # select kv according to routing index
        b, mn, ws2, c = image.size()
        topw_kv = torch.gather(image.view(b, 1, mn, ws2, c).expand(-1, index.shape[1], -1, -1, -1), # (b Nq mn ws^2 c) 
                                dim=2,
                                index=index.view(b, index.shape[1], self.topw, 1, 1).expand(-1, -1, -1, ws2, c) # (b Nq topw ws^2 c)
                               )
        return topw_kv 


class TopwCrossAttention(nn.Module):

    def __init__(self, dim_q, dim_kv, dim_out, heads=8, dim_heads=32, topw=4, winsdow_size=8, factor=2, dropout=0):
        super().__init__()
        inner_dim = dim_heads * heads
        project_out = not (heads == 1 and dim_heads == dim_q)

        self.heads = heads
        self.scale = dim_heads ** -0.5
        self.window_size = winsdow_size
        self.topw = topw
        self.scalei = dim_kv  ** -0.5

        self.key_pos_embed = nn.Parameter(torch.randn(1, topw*winsdow_size*winsdow_size, inner_dim))

        self.from_image = nn.Conv2d(dim_kv, inner_dim, kernel_size=factor, stride=factor)
        self.image_window_routing = WindowRouting(dim=inner_dim, topw=topw)
        self.image_proposals = ProposalGather(topw=topw)

        self.attend = nn.Softmax(dim=-1)
        self.normq = nn.LayerNorm(dim_q)
        self.normkv = nn.LayerNorm(inner_dim)
        self.to_q = nn.Linear(dim_q, inner_dim, bias = False)
        self.to_k = nn.Linear(inner_dim, inner_dim, bias = False)
        self.to_v = nn.Linear(inner_dim, inner_dim, bias = False)
        self.recover_q = nn.Linear(inner_dim, dim_q, bias = False)

        self.to_out = MLP(dim_q, dim_out, dropout=dropout) if project_out else nn.Identity()

    def forward(self, query, image):
        # window self-attention weighting
        imagec= self.from_image(image)
        H, W = imagec.shape[2], imagec.shape[3]

        assert H%self.window_size == 0 and W%self.window_size == 0, "The image feature map should be divisible by the window size."

        imagec = rearrange(imagec, 'b c (h m) (w n) -> (b m n) (h w) c', h=self.window_size, w=self.window_size) # (bmn hw c)
        imagecor = torch.matmul(imagec, imagec.transpose(-1, -2)) * self.scalei # (bmn hw hw)
        imagecor = softmax_one(imagecor, dim=-1)
        imagecor = torch.sum(imagecor, dim=1) # (bmn hw)
        imagecor = softmax_one(imagecor, dim=-1).unsqueeze(-1) # (bmn hw 1)
        imagew = imagec * imagecor # (bmn hw c)
        imagew = torch.sum(imagew, dim=1) # (bmn c)
        imagew = rearrange(imagew, '(b m n) c -> b (m n) c', m=H//self.window_size, n= W//self.window_size) # (b mn c)

        # select self.window_number windows for interaction
        q = self.to_q(self.normq(query)) # (b Nq c)
        topw_index = self.image_window_routing(q, imagew) # (b Nq topw)
        imagec = rearrange(imagec, '(b m n) p c -> b (m n) p c', m=H//self.window_size, n= W//self.window_size) # (b mn ws^2 c)
        image_proposals = self.image_proposals(topw_index, imagec) # (b Nq topw ws^2 c)
        query_num = q.shape[1]
        image_proposals = rearrange(image_proposals, 'b q w p c -> (b q) (w p) c') # (b*Nq topw*ws^2 c)
        image_proposals = image_proposals + self.key_pos_embed[:, :image_proposals.size(1), :]

        q = rearrange(q, 'b (q a) (g c) -> (b q) g a c', a=1, g=self.heads) # (b*Nq g 1 d)
        kv = self.normkv(image_proposals)
        k = self.to_k(kv)
        k = rearrange(k, 'b n (g c) -> b g n c', g=self.heads) # (b*Nq g topw*ws^2 d)
        v = self.to_v(kv)
        v = rearrange(v, 'b n (g c) -> b g n c', g=self.heads) # (b*Nq g topw*ws^2 d)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale # (b*Nq g 1 topw*ws^2)
        attn = self.attend(dots)
        out = torch.matmul(attn, v).squeeze(2) # (b*Nq g d)
        out = rearrange(out, '(b q) g c -> b q (g c)', q=query_num) # (b Nq c)
        out = self.recover_q(out) + query
        out =  out + self.to_out(out)
        return out


def build_attention_mask(text_mask, len_q, len_k):

    attn_mask = torch.zeros((len_q, len_k)).repeat(text_mask.size(0), 1, 1).to(text_mask.device) # [b nq nk]
    inf = torch.zeros((len_q, len_k)).fill_(float("-inf")).repeat(text_mask.size(0), 1, 1).to(text_mask.device) # [b nq nk]
    text_mask = text_mask.unsqueeze(1).expand(-1, len_q, -1)
    attn_mask = torch.where(text_mask>0, attn_mask, inf) # [b nq nk]

    return attn_mask


class ObjectAlignment(nn.Module):

    def __init__(self, dim_image, dim_text, dim_query, dim_alignment, n_token=25, text_ksize=5, num_heads=8, dim_heads=32, window_size=8, window_num=4, dropout=0.1):
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads

        self.text_offset1 = nn.Linear(n_token, 2*text_ksize, bias = False)
        self.text_offset2 = nn.Linear(dim_text, 2*text_ksize, bias = False)

        self.text_deformable1d = DeformConv2d(in_channels=dim_text, out_channels=dim_alignment, kernel_size=(text_ksize, 1), stride=(1, 1), padding=((text_ksize-1)//2, 0))

        #self.text_to_object = CrossAttention(dim_q=dim_query, dim_kv=dim_alignment, dim_out=dim_alignment, heads=num_heads, dim_heads=dim_heads)
        
        self.text_to_object = MaskedCrossAttentionBlock(dim_q=dim_query, dim_kv=dim_alignment, dim_out=dim_alignment, heads=num_heads, layers=1, dropout=dropout)

        self.image_to_object = TopwCrossAttention(dim_q=dim_query, dim_kv=dim_image, dim_out=dim_alignment, heads=num_heads, dim_heads=dim_heads, topw=window_num, winsdow_size=window_size, dropout=dropout)

        self.to_query = MLP(2*dim_alignment, dim_query, dropout=dropout)

        #self.object_to_text = CrossAttention(dim_q=dim_text, dim_kv=dim_query, dim_out=dim_text, heads=num_heads, dim_heads=dim_heads)

        self.object_to_text = MaskedCrossAttentionBlock(dim_q=dim_text, dim_kv=dim_query, dim_out=dim_text, heads=num_heads, layers=1, dropout=dropout)

        #self.object_to_image = CrossAttention(dim_q=dim_image, dim_kv=dim_query, dim_out=dim_image, heads=num_heads, dim_heads=dim_heads)

        self.object_to_image = MaskedCrossAttentionBlock(dim_q=dim_image, dim_kv=dim_query, dim_out=dim_image, heads=num_heads, layers=1, dropout=dropout)
     
        self.scalet = dim_text  ** -0.5

    def forward(self, image, text, query, attn_mask_tkey=None):
        """ image:(b c1 H W), text:(b L c2), query:(b q c3), text_mask:(b L) """
        # for text-level object interaction 
        textcor = torch.matmul(text, text.transpose(-1, -2)) * self.scalet # [b L L]
        textcor = softmax_one(textcor, dim=-1) # [b L L]
        offset1 = self.text_offset1(textcor) # [b L 2*ksize]
        offset2 = self.text_offset2(text) # [b L 2*ksize]
        offset = (offset1+ offset2).permute(0, 2, 1).unsqueeze(-1) # [b 2*ksize L 1]

        textc = text.permute(0, 2, 1).unsqueeze(-1) # [b c2 L 1]
        textc = self.text_deformable1d(textc, offset).squeeze(-1) # [b c L]
        textc =  textc.permute(0, 2, 1) # [b L c]

        query1 = self.text_to_object(query.permute(1, 0, 2), textc.permute(1, 0, 2), attn_mask_tkey) # [q b c]
        query1 = query1.permute(1, 0, 2)  # [b q c]

        # for image-level object interaction
        query2 = self.image_to_object(query, image)

        query = torch.cat((query1, query2), dim=-1) # [b q 2c]
        query = self.to_query(query) # [b q c]

        text = self.object_to_text(text.permute(1, 0, 2), query.permute(1, 0, 2))
        text = text.permute(1, 0, 2)  # [b q c]

        H, W = image.shape[2], image.shape[3]
        image = rearrange(image, 'b c H W -> b (H W) c')
        image = self.object_to_image(image.permute(1, 0, 2), query.permute(1, 0, 2))
        image = image.permute(1, 0, 2)  # [b HW c]
        image = rearrange(image, 'b (H W) c -> b c H W', H=H, W=W)

        return image, text, query

# -------------------------------------------------------------------------------------------------------------------------------

class OA_ablation(nn.Module):

    def __init__(self, dim_image, dim_text, dim_query, dim_alignment, n_token=25, text_ksize=5, num_heads=8, dim_heads=32, window_size=8, window_num=4, flags=[0, 0, 0, 0]):
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads

        self.flag_t2o, self.flag_o2t, self.flag_i2o, self.flag_o2i = flags[0], flags[1], flags[2], flags[3]

        if self.flag_t2o:
            self.text_offset1 = nn.Linear(n_token, 2*text_ksize, bias = False)
            self.text_offset2 = nn.Linear(dim_text, 2*text_ksize, bias = False)
            self.text_deformable1d = DeformConv2d(in_channels=dim_text, out_channels=dim_alignment, kernel_size=(text_ksize, 1), stride=(1, 1), padding=((text_ksize-1)//2, 0))
            self.text_to_object = MaskedCrossAttentionBlock(dim_q=dim_query, dim_kv=dim_alignment, dim_out=dim_alignment, heads=num_heads, layers=1)
            if self.flag_i2o == 0:
                self.to_query = MLP(dim_alignment, dim_query)

        if self.flag_i2o:
            self.image_to_object = TopwCrossAttention(dim_q=dim_query, dim_kv=dim_image, dim_out=dim_alignment, heads=num_heads, dim_heads=dim_heads, topw=window_num, winsdow_size=window_size)
            if self.flag_t2o == 0:
                self.to_query = MLP(dim_alignment, dim_query)
            else:
                self.to_query = MLP(2*dim_alignment, dim_query)

        if self.flag_o2t:
            self.object_to_text = MaskedCrossAttentionBlock(dim_q=dim_text, dim_kv=dim_query, dim_out=dim_text, heads=num_heads, layers=1)
        if self.flag_o2i:
            self.object_to_image = MaskedCrossAttentionBlock(dim_q=dim_image, dim_kv=dim_query, dim_out=dim_image, heads=num_heads, layers=1)
     
        self.scalet = dim_text  ** -0.5

    def forward(self, image, text, query, attn_mask_tkey=None):
        """ image:(b c1 H W), text:(b L c2), query:(b q c3), text_mask:(b L) """
        # for text-level object interaction 
        if self.flag_t2o:
            textcor = torch.matmul(text, text.transpose(-1, -2)) * self.scalet # [b L L]
            textcor = softmax_one(textcor, dim=-1) # [b L L]
            offset1 = self.text_offset1(textcor) # [b L 2*ksize]
            offset2 = self.text_offset2(text) # [b L 2*ksize]
            offset = (offset1+ offset2).permute(0, 2, 1).unsqueeze(-1) # [b 2*ksize L 1]

            textc = text.permute(0, 2, 1).unsqueeze(-1) # [b c2 L 1]
            textc = self.text_deformable1d(textc, offset).squeeze(-1) # [b c L]
            textc =  textc.permute(0, 2, 1) # [b L c]

            query1 = self.text_to_object(query.permute(1, 0, 2), textc.permute(1, 0, 2), attn_mask_tkey) # [q b c]
            query1 = query1.permute(1, 0, 2)  # [b q c]

        # for image-level object interaction
        if self.flag_i2o:
            query2 = self.image_to_object(query, image)

        if self.flag_t2o:
            if self.flag_i2o:
                query = torch.cat((query1, query2), dim=-1) # [b q 2c]
                query = self.to_query(query) # [b q c]
            else:
                query = query1
                query = self.to_query(query) # [b q c]
        else:
            if self.flag_i2o:
                query = query2
                query = self.to_query(query) # [b q c]
    
        if self.flag_o2t:
            text = self.object_to_text(text.permute(1, 0, 2), query.permute(1, 0, 2))
            text = text.permute(1, 0, 2)  # [b q c]

        if self.flag_o2i:
            H, W = image.shape[2], image.shape[3]
            image = rearrange(image, 'b c H W -> b (H W) c')
            image = self.object_to_image(image.permute(1, 0, 2), query.permute(1, 0, 2))
            image = image.permute(1, 0, 2)  # [b HW c]
            image = rearrange(image, 'b (H W) c -> b c H W', H=H, W=W)

        return image, text, query




        
 







