# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from .text_encoder import TextEncoder
from einops import rearrange


def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()


def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU'):
    layers = []
    layers.append(ConvBatchNorm(in_channels, out_channels, activation))
    for _ in range(nb_Conv - 1):
        layers.append(ConvBatchNorm(out_channels, out_channels, activation))
    return nn.Sequential(*layers)


class ConvBatchNorm(nn.Module):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(ConvBatchNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)


class DownBlock(nn.Module):
    """Downscaling with maxpool convolution"""

    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(DownBlock, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x):
        out = self.maxpool(x)
        return self.nConvs(out)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
    

class Upblock(nn.Module):
    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x, skip_x):
        up = self.up(x)
        x = torch.cat([skip_x, up], dim=1)  # dim 1 is the channel dimension
        return self.nConvs(x)


# -------------------------------------------------------------------------------------------------------------------

from .alignment import ObjectAlignment, build_attention_mask, Transformer
from models.DAM.decoder import  OrderMasksDecoder
from models.DAM.loss import DynamicMasksLoss_w_Matching

class DAM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_channels = config.n_channels
        self.basedim = config.base_channel
        dim = 192
        self.query_num = config.num_query
        self.query_dim = dim
        self.text_num = config.token_length
        self.text_dim = dim
        self.align_dim = dim
        self.align_ksize = 5
        self.align_heads = 8 #8 
        self.align_dim_head = 32 #32
        self.align_window_size = config.align_window_size
        self.w_merged = 0.0
        self.w_masks = 0.8
        self.w_class = 0.05
        self.w_num = 0.05 
        self.w_om = 0.1
        self.w_bce = 0.5
        self.w_dc = 0.5


        self.object_query = nn.Parameter(torch.randn(1, self.query_num, self.query_dim))
        self.align_layer1 = ObjectAlignment(self.basedim * 2, self.text_dim, self.query_dim, self.align_dim, self.text_num, self.align_ksize, self.align_heads, self.align_dim_head, self.align_window_size, 8)#8
        self.align_layer2 = ObjectAlignment(self.basedim * 4, self.text_dim, self.query_dim, self.align_dim, self.text_num, self.align_ksize, self.align_heads, self.align_dim_head, self.align_window_size, 4)#4
        self.align_layer3 = ObjectAlignment(self.basedim * 8, self.text_dim, self.query_dim, self.align_dim, self.text_num, self.align_ksize, self.align_heads, self.align_dim_head, self.align_window_size, 2)#2
        self.align_layer4 = ObjectAlignment(self.basedim * 8, self.text_dim, self.query_dim, self.align_dim, self.text_num, self.align_ksize, self.align_heads, self.align_dim_head, self.align_window_size, 1)#1
           
        self.inc = ConvBatchNorm(self.n_channels, self.basedim)

        self.down1 = DownBlock(self.basedim, self.basedim * 2, nb_Conv=2)
        self.down2 = DownBlock(self.basedim * 2, self.basedim * 4, nb_Conv=2)
        self.down3 = DownBlock(self.basedim * 4, self.basedim * 8, nb_Conv=2)
        self.down4 = DownBlock(self.basedim * 8, self.basedim * 8, nb_Conv=2)

        self.up4 = Upblock(self.basedim * 16, self.basedim * 4, nb_Conv=2)
        self.up3 = Upblock(self.basedim * 8, self.basedim * 2, nb_Conv=2)
        self.up2 = Upblock(self.basedim * 4, self.basedim, nb_Conv=2)
        self.up1 = Upblock(self.basedim * 2, self.basedim, nb_Conv=2)
        
        self.text_encoder =  TextEncoder(in_channels=768, out_channels=self.text_dim, text_length=config.token_length, text_layers=4, text_heads=8, pre_pro=True)

        self.decoder = OrderMasksDecoder(image_dim=self.basedim, text_dim=self.text_dim, query_dim=self.query_dim, align_dim=self.align_dim, text_num=self.text_num, query_num=self.query_num, image_size=config.img_size)
        self.criterion = DynamicMasksLoss_w_Matching(num_query=self.query_num, w_merged=self.w_merged, w_masks=self.w_masks, w_class=self.w_class, w_num=self.w_num, w_bce=self.w_bce, w_dc=self.w_dc)

        self.res_text = True

        self.vit_depth=1
        num_patches = (config.img_size//(2**4))**2
        self.pos_embedding_image = nn.Parameter(torch.randn(1, num_patches, self.basedim * 8))
        self.transformers = nn.ModuleList()
        for i in range(self.vit_depth):
            self.transformers.append(Transformer(self.basedim * 8, 1, 8, 32, 1024, 0.1))
        

    def forward(self, batched_inputs, device="cuda", gflops=False):

        x = batched_inputs['image'].to(dtype = torch.float32, device=device)
        text = batched_inputs['lang_token'].to(device=device)
        text_mask = batched_inputs['lang_mask'].to(device=device)

        query = self.object_query.repeat(text.shape[0], 1, 1) # [b Nq c]
        text_feat = self.text_encoder(text, text_mask)

        attn_mask_tkey = build_attention_mask(text_mask, query.size(1), text_feat.size(1))

        x1 = self.inc(x)  # x1 [4, 64, 224, 224]

        x2 = self.down1(x1)
        resx, restext, query = self.align_layer1(x2, text_feat, query, attn_mask_tkey)
        x2 = x2+resx
        text_feat = text_feat + restext if self.res_text else restext

        x3 = self.down2(x2)
        resx, restext, query = self.align_layer2(x3, text_feat, query, attn_mask_tkey)
        x3 = x3 + resx
        text_feat = text_feat + restext if self.res_text else restext

        x4 = self.down3(x3)
        resx, restext, query = self.align_layer3(x4, text_feat, query, attn_mask_tkey)
        x4 = x4 + resx
        text_feat = text_feat + restext if self.res_text else restext
        
        x5 = self.down4(x4)
        resx, restext, query = self.align_layer4(x5, text_feat, query, attn_mask_tkey)
        x5 = x5 + resx
        text_feat = text_feat + restext if self.res_text else restext 

        #------------------------------------------------------
        h, w = x5.shape[2], x5.shape[3]
        x5 = rearrange(x5, 'b c h w -> b (h w) c')
        x5 = x5 + self.pos_embedding_image[:, :h*w]
        for i_layer in range(self.vit_depth):
            x5 = self.transformers[i_layer](x5)
        x5 = rearrange(x5, 'b (h w) c -> b c h w', h=h, w=w)
        #------------------------------------------------------
        
        x = self.up4(x5, x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)

        merged_pred, all_masks_sorted, query_pro_sorted, predict_masks_num, query, text, all_masks, query_pro = self.decoder(x, text_feat, query) 

        if gflops:
            return merged_pred
        else:
            masks_gt = batched_inputs['mask_instances'].to(dtype = torch.float32, device=device) # (b q h w)
            merged_gt = batched_inputs['mask_merged'].to(dtype = torch.float32, device=device) # (b h w)
            cquery_gt = batched_inputs['mask_existing'].to(dtype = torch.long, device=device) # (b q)
            obnum_gt = batched_inputs['class_num'].to(dtype = torch.long, device=device) # (b)
            combined_gt = batched_inputs['mask_combined'].to(dtype = torch.long, device=device) # (b h w)

            text_label = batched_inputs['text_label'].to(dtype = torch.long, device=device) # (b l)
            loss_dict = self.criterion(merged_pred, merged_gt, all_masks, masks_gt, query_pro, cquery_gt, predict_masks_num, obnum_gt, query, text, text_label)

            return {"predict": merged_pred, "existing":None, "loss":loss_dict["total_loss"], "loss_dict": loss_dict,
                    "pred_masks": all_masks_sorted, "pred_cquery": query_pro_sorted, "pred_num":predict_masks_num*self.query_num, "query_num": self.query_num}








