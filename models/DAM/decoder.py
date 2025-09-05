""" Components of the dynamic masks decoder of DAM """
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# ------------------------------------------------ for dynamic masks decoder ------------------------------------------------
class GlobalizationSimple(nn.Module):

    def __init__(self, text_num, text_dim):
        super().__init__()
        self.combine =  nn.Linear(2*text_dim, text_dim, bias = False)  
        self.text_num = text_num

    def forward(self, text):
        text = text.permute(0, 2, 1) # (b c t)
        avg = F.avg_pool1d(text, kernel_size=self.text_num, stride=self.text_num)
        avg = avg.permute(0, 2, 1) # (b 1 c)
        max = F.max_pool1d(text, kernel_size=self.text_num, stride=self.text_num)
        max = max.permute(0, 2, 1) # (b 1 c)
        text = torch.cat((avg, max), dim=-1) # (b 1 2c)
        text = self.combine(text) # (b 1 c)
        return text


class Globalization(nn.Module):

    def __init__(self, text_num, text_dim):
        super().__init__()
        self.learnable =  nn.Linear(text_num, 1, bias = False)
        self.combine =  nn.Linear(2*text_dim, text_dim, bias = False)
        self.softmax = nn.Softmax(dim=-1)
        self.scale = text_dim ** -0.5

    def forward(self, text):
        attnw = torch.matmul(text, text.transpose(-1, -2)) * self.scale # (b t t)
        attnw = torch.sum(attnw, dim=1)
        attnw = self.softmax(attnw).unsqueeze(-1) # (b t 1)
        text1 = text * attnw # (b t c)
        text1 = torch.sum(text1, dim=1, keepdim=True) # (b 1 c)
        text2 = self.learnable(rearrange(text, 'b t c -> b c t')) # (b c 1)
        text = torch.cat((text1, rearrange(text2, 'b c t -> b t c')), dim=-1) # (b 1 2c)
        text = self.combine(text) # (b 1 c)
        return text


class GlobalizationLearn(nn.Module):

    def __init__(self, text_num, text_dim):
        super().__init__()
        #self.learnable =  nn.Linear(text_num, 1, bias = False)
        self.learnable =  MLP(text_num, 1)

    def forward(self, text):
        text = self.learnable(rearrange(text, 'b t c -> b c t')) # (b c 1)
        text = rearrange(text, 'b c t -> b t c')
        return text
    

def TextGrouping(text, query):
    with torch.no_grad():
        text_group = torch.matmul(text, query.transpose(-1, -2)) # (b t q)
        idx_cluster = text_group.argmax(dim=-1) # (b t)
    return idx_cluster


class MLP(nn.Module):

    def __init__(self, in_channels, out_channels, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(in_channels)
        self.net = nn.Sequential(
            nn.Linear(in_channels, in_channels//2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(in_channels//2, out_channels),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.norm(x)
        x = self.net(x)
        return x
    

class OrderQuery(nn.Module):

    def __init__(self, smooth=0.001):
        super().__init__()
        self.smooth = smooth

    def forward(self, query, text):
        # initialize the order
        b, q, _ = query.shape
        t = text.shape[1]
        
        idx_cluster = TextGrouping(text, query) # b t
        idx_batch = torch.arange(b, device=query.device)[:, None] # (b 1)
        idx = idx_cluster + idx_batch * (q) # (b t)

        init_order = query.new_zeros(b, q, 1) + q # (b q 1)
        off_order = torch.arange(t, 0, -1, device=query.device)[None, :, None].repeat(b, 1, 1).float() # (b t 1)
        off_count = query.new_ones((b, t, 1))
        q_off_order = query.new_zeros(b*q, 1)
        q_off_order.index_add_(dim=0, index=idx.reshape(b*t), source=off_order.reshape(b*t, 1))
        q_off_count = query.new_zeros(b*q, 1)
        q_off_count.index_add_(dim=0, index=idx.reshape(b*t), source=off_count.reshape(b*t, 1))
        q_off_order = q_off_order/(q_off_count + self.smooth)
        q_off_order = rearrange(q_off_order, '(b q) a -> b q a', b=b) # (b q 1)
        queryorder = init_order - q_off_order 
        
        # The ascending sorting result is consistent with the order in which the objects appear in the text
        _, sorted_indices = torch.sort(queryorder.squeeze(-1), dim=1, descending=False)
      
        return sorted_indices
    

class DynamicMasksDecoder(nn.Module):

    def __init__(self, image_dim, text_dim, query_dim, align_dim, text_num):
        super().__init__()
        self.globalization = Globalization(text_num=text_num, text_dim=align_dim)
        self.uni_image = nn.Linear(image_dim, align_dim, bias = False)
        self.uni_text = nn.Linear(text_dim, align_dim, bias = False)
        self.uni_query = nn.Linear(query_dim, align_dim, bias = False)
        self.order_query = OrderQuery()
        self.scale = align_dim ** -0.5
        self.sigmoid = nn.Sigmoid()
        self.classifier = MLP(in_channels=query_dim, out_channels=2)
        self.regressor =  MLP(in_channels=align_dim, out_channels=1)

    def forward(self, image, text, query):
        """ image:(b c1 H W), text:(b t c2), query:(b q c3) """
        _, _, H, W = image.shape
        image = rearrange(image, 'b c H W -> b (H W) c')
        image = self.uni_image(image)
        text = self.uni_text(text)
        query = self.uni_query(query)
        # locate the merged masks
        text_global = self.globalization(text) # (b 1 c)
        merged_mask = torch.matmul(text_global, image.transpose(-1, -2)) * self.scale # (b 1 HW)
        merged_mask = rearrange(merged_mask, 'b t (h w) -> b t h w', h=H, w=W) # (b 1 H W)
        # use merged masks to highlight the ROI before obtain object-level masks
        #image = image * self.sigmoid(rearrange(merged_mask, 'b t h w -> b (h w) t')) # (b HW c)
        all_masks = torch.matmul(query, image.transpose(-1, -2)) * self.scale # (b q HW)
        all_masks = rearrange(all_masks, 'b t (h w) -> b t h w', h=H, w=W) # (b q H W)
        # use global text to highlight the related queries before determining the semantics of the queries
        attn_query = torch.matmul(text_global, query.transpose(-1, -2)) * self.scale # (b 1 q)
        querya = query * self.sigmoid(attn_query.permute(0, 2, 1))
        query_order = self.order_query(querya, text) # (b q)
        all_masks_sorted = all_masks.gather(1, query_order.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W))
        query_pro = self.classifier(querya) # (b q 2)
        query_pro_sorted =  query_pro.gather(1, query_order.unsqueeze(-1).expand(-1, -1, 2))
        predict_masks_num = self.regressor(text_global).squeeze(-1).squeeze(-1) # b
        predict_masks_num = self.sigmoid(predict_masks_num)

        return merged_mask, all_masks_sorted, query_pro_sorted, predict_masks_num

# ----------------------------------------------------------------------------------------------------------------------
class ExistencePredictor(nn.Module):

    def __init__(self, in_channels, out_channels=2, stride=8):
        super().__init__()
        self.max_pool = nn.MaxPool2d(stride, stride)
        self.fc = nn.Linear(in_channels, out_channels, bias = False)

    def forward(self, x):
        """x:(b q h w) """
        x = self.max_pool(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = self.fc(x) # (b q 2)
        return x


class ModulationConvolution(nn.Module):

    def __init__(self, query_dim, image_dim, kernel_size=3):
        super().__init__()
        self.image_dim = image_dim
        self.kernel_size = kernel_size
        
        self.to_factor = nn.Linear(query_dim, image_dim* kernel_size * kernel_size)

    def forward(self, query, image):
        """ image:(b c1 H W), query:(b q c3) """
        b, q, _ = query.shape
        _, _, h, w = image.shape
        query =  rearrange(query, 'b (q n) c -> (b q) n c', n=1)
        padding =  self.kernel_size // 2
        stride = 1
        kernel = self.to_factor(query)
        kernel = kernel.reshape(b*q, self.image_dim, self.kernel_size, self.kernel_size) # (bq c k k)
        image = image.reshape(1, b*self.image_dim, h, w) #(1 bc h w)
        mask = F.conv2d(image, kernel, padding=padding, stride=stride, groups=b) # (1 bq h w)
        mask = mask.reshape(b, q, h, w)
        return mask


class DynamicMasksDecoder3(nn.Module):

    def __init__(self, image_dim, text_dim, query_dim, align_dim, text_num, pre=False, query_num=6, image_size=256):
        super().__init__()
    
        self.globalization = Globalization(text_num=text_num, text_dim=align_dim)
        self.uni_image = nn.Linear(image_dim, align_dim, bias = False)
        self.uni_text = nn.Linear(text_dim, align_dim, bias = False)
        self.uni_query = nn.Linear(query_dim, align_dim, bias = False)
        self.order_query = OrderQuery()
        self.scale = align_dim ** -0.5
        self.sigmoid = nn.Sigmoid()

        self.class_version = 2
        if self.class_version == 1:
            self.classifier = MLP(in_channels=query_dim, out_channels=2)
        else:
            self.classifier = ExistencePredictor((image_size//8)**2, 2, 8)

        self.regressor =  MLP(in_channels=align_dim, out_channels=1)

        self.merge = nn.Conv2d(query_num, 2, kernel_size=3, padding=1)

        self.pre = pre
        if pre:
            self.to_masks = ModulationConvolution(query_dim, image_dim, 3)
        else:
            self.to_masks = ModulationConvolution(align_dim, align_dim, 3)

    def forward(self, image, text, query):
        """ image:(b c1 H W), text:(b t c2), query:(b q c3) """
        _, _, H, W = image.shape

        if self.pre:
            all_masks = self.to_masks(query, image) # (b q h w)

        image = rearrange(image, 'b c H W -> b (H W) c')
        image = self.uni_image(image)
        text = self.uni_text(text)
        query = self.uni_query(query)

        # obtain mask for each object query
        all_masks1 = torch.matmul(query, image.transpose(-1, -2)) * self.scale # (b q HW)
        all_masks1 = rearrange(all_masks1, 'b t (h w) -> b t h w', h=H, w=W) # (b q H W)
        
        if not self.pre:
            image = rearrange(image, 'b (H W) c -> b c H W', H=H)
            all_masks =  self.to_masks(query, image)

        all_masks =  all_masks + all_masks1 # (b q h w)

        merged_mask = self.merge(all_masks) # (b 2 H W)

        # use global text to highlight the related queries before determining the semantics of the queries
        text_global = self.globalization(text) # (b 1 c)
        attn_query = torch.matmul(text_global, query.transpose(-1, -2)) * self.scale # (b 1 q)
        querya = query * self.sigmoid(attn_query.permute(0, 2, 1))
        query_order = self.order_query(querya, text) # (b q)
        all_masks_sorted = all_masks.gather(1, query_order.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)) # b q h w
        all_masks_sorted = torch.cat((all_masks_sorted[:, -1:, :, :], all_masks_sorted[:, :-1, :, :]), dim=1)

        #------------- version1 or version2 ---------------------------
        if self.class_version == 1:
            query_pro = self.classifier(querya) # (b q 2)
            query_pro_sorted =  query_pro.gather(1, query_order.unsqueeze(-1).expand(-1, -1, 2))
            query_pro_sorted = torch.cat((query_pro_sorted[:, -1:, :], query_pro_sorted[:, :-1, :]), dim=1)
        else:
            query_pro_sorted =  self.classifier(all_masks_sorted) # b q 2

        predict_masks_num = self.regressor(text_global).squeeze(-1).squeeze(-1) # b
        predict_masks_num = self.sigmoid(predict_masks_num)

        # -------- add -∞ to all_masks_sorted or not -----------------
        attn_mask = torch.zeros((all_masks.shape[1], all_masks.shape[2], all_masks.shape[3])).repeat(all_masks.shape[0], 1, 1, 1).to(all_masks.device) + 0 # [b q h w]
        inf = torch.zeros((all_masks.shape[1], all_masks.shape[2], all_masks.shape[3])).fill_(float("-inf")).repeat(all_masks.shape[0], 1, 1, 1).to(all_masks.device)
        #inf = torch.zeros((all_masks.shape[1], all_masks.shape[2], all_masks.shape[3])).repeat(all_masks.shape[0], 1, 1, 1).to(all_masks.device) - 1
        inf[:, 0, :, :] = 0
        exist = F.softmax(query_pro_sorted, dim=-1)[:, :, 1:] # b q 1 
        exist = exist.unsqueeze(-1).expand(-1, -1, all_masks.shape[2], all_masks.shape[3]) # b q h w
        attn_mask = torch.where(exist>0.5, attn_mask, inf) # (b q h w)

        all_masks_sorted =  all_masks_sorted + attn_mask
        #all_masks_sorted =  all_masks_sorted * exist

        return merged_mask, all_masks_sorted, query_pro_sorted, predict_masks_num


# ====================================================================================================================
class ExistencePredictor2(nn.Module):

    def __init__(self, in_channels, out_channels=2, stride=8):
        super().__init__()
        self.max_pool = nn.MaxPool2d(stride, stride)
        self.fc = nn.Linear(in_channels, out_channels, bias = False)

    def forward(self, x, query):
        """x:(b q h w) q:(b q d)"""
        x = self.max_pool(x)
        x = x.reshape(x.shape[0], x.shape[1], -1) # (b q hw)
        xq = torch.cat((x, query), dim=-1)
        x = self.fc(xq) # (b q 2)
        return x


def obtain_text_level_object(query, text):
    text_group = torch.matmul(query, text.transpose(-1, -2)) * (query.shape[2] ** (-0.5)) # (b q t)
    text_group = F.softmax(text_group, dim=1) # (b q t)
    text_query = torch.matmul(text_group, text) * (text.shape[1] ** (-0.5)) # (b q d)
    return text_query


class QClassifier(nn.Module):

    def __init__(self, in_channels, out_channels, factor=0.5, dropout=0.):
        super().__init__()
        #self.mlp = MLP2(in_channels, out_channels, factor, dropout)
        self.mlp = nn.Linear(in_channels, out_channels)
        self.maxpool = nn.AdaptiveMaxPool2d(output_size=(1, 1))
    def forward(self, query, image):
        """query:(b q d), image: (b k d)"""
        b, q, d = query.shape
        pooled_results = []
        for i in range(q):
            qi = query[:, i, :].unsqueeze(-1).unsqueeze(-1)
            qimage =  qi * image
            qimage = self.maxpool(qimage).squeeze(-1).squeeze(-1).unsqueeze(1) # b 1 d
            pooled_results.append(qimage)
        pooled_results = torch.cat(pooled_results, dim=1)
        pooled_results = self.mlp(pooled_results) # b q 2 
        return pooled_results

class OrderMasksDecoder(nn.Module):

    def __init__(self, image_dim, text_dim, query_dim, align_dim, text_num, query_num=6, image_size=256, dropout=0.1):
        super().__init__()
    
        self.globalization = Globalization(text_num=text_num, text_dim=align_dim)
        self.uni_image = nn.Linear(image_dim, align_dim, bias = False)
        self.uni_text = nn.Linear(text_dim, align_dim, bias = False)
        self.uni_query = nn.Linear(query_dim, align_dim, bias = False)
        self.order_query = OrderQuery()
        self.scale = align_dim ** -0.5
        self.sigmoid = nn.Sigmoid()

        self.class_version = 3 
        if self.class_version == 1:
            self.classifier = MLP(in_channels=query_dim, out_channels=2, dropout=dropout)
        elif self.class_version == 3:
            self.classifier = ExistencePredictor2((image_size//8)**2 + query_dim, 2, 8)
        elif self.class_version == 4:
            self.classifier = QClassifier(in_channels=query_dim, out_channels=2, factor=0.5, dropout=dropout)
        else:
            self.classifier = ExistencePredictor((image_size//8)**2, 2, 8)

        self.regressor =  MLP(in_channels=align_dim, out_channels=1, dropout=dropout)

        self.merge = nn.Conv2d(align_dim, 2,  kernel_size=1, stride=1)

        self.to_masks = ModulationConvolution(align_dim, align_dim, 1)



    def forward(self, image0, text, query):
        """ image:(b c1 H W), text:(b t c2), query:(b q c3) """
        _, _, H, W = image0.shape

        image = rearrange(image0, 'b c H W -> b (H W) c')
        image = self.uni_image(image)
        text = self.uni_text(text)
        query = self.uni_query(query)

        text_query = 0.5*obtain_text_level_object(query, text) + 0.5*query

         
        image = rearrange(image, 'b (H W) c -> b c H W', H=H)
       
        all_masks =  self.to_masks(text_query, image) 
 
        merged_mask = self.merge(image)    

        text_global = self.globalization(text) # (b 1 c)

        #------------- version1 or version2 ---------------------------
        if self.class_version == 1:
            query_pro = self.classifier(text_query) # (b q 2)
        elif self.class_version == 3:
            query_pro =  self.classifier(all_masks, text_query)
        elif self.class_version == 4:
            query_pro = self.classifier(text_query, image)
        else:
            query_pro =  self.classifier(all_masks) # b q 2

        predict_masks_num = self.regressor(text_global).squeeze(-1).squeeze(-1) # b
        predict_masks_num = self.sigmoid(predict_masks_num)


        return merged_mask, all_masks, query_pro, predict_masks_num, query ,text, all_masks, query_pro