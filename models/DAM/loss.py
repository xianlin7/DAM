""" Components of the loss computation of DAM """
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

class DiceLoss(nn.Module):

    def __init__(self, smooth=1.):
        super().__init__()

        self.apply_nonlin = nn.Sigmoid()
        self.smooth = smooth

    def forward(self, pred, target):
        ''' pred:(b h w), target:(b h w) '''
        pred = torch.sigmoid(pred) # b h w
        pred = pred.contiguous().view(pred.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        intersection = torch.sum(pred * target, dim=-1) # (b)
        union = torch.sum(pred, dim=-1) + torch.sum(target, dim=-1)
        dice = (2 * intersection + self.smooth) / (union + self.smooth) # (b)
        loss = 1 - dice
        return loss.mean()


class TopkDiceLoss(nn.Module):

    def __init__(self, smooth=1.):
        super().__init__()

        self.smooth = smooth
        self.dice_loss = DiceLoss(self.smooth)
    
    def forward(self, pred, target, object_num):
        ''' pred:(b q h w), target:(b q h w), object_num:(b) '''
        B = pred.shape[0]
        total_loss = 0
        for b in range(B):
            numb = object_num[b]
            lossb = 0
            predb = pred[b, :numb, :, :]
            targetb = target[b, :numb, :, :]
            lossb = self.dice_loss(predb, targetb)
            total_loss = total_loss + lossb
        total_loss = total_loss/B
        return total_loss


class TopkCELoss(nn.Module):

    def __init__(self):
        super().__init__()

        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, pred, target, object_num):
        ''' pred:(b q 2), target:(b q), object_num:(b) '''
        B = pred.shape[0]
        total_loss = 0
        for b in range(B):
            numb = object_num[b]
            predb = pred[b, :numb, :]
            targetb = target[b, :numb]
            lossb = self.ce_loss(predb, targetb.long()) 
            total_loss = total_loss + lossb
        total_loss = total_loss/B
        return total_loss


class TopkBCELoss(nn.Module):

    def __init__(self):
        super().__init__()

        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, pred, target, object_num):
        ''' pred:(b q h w), target:(b q h w), object_num:(b) '''
        B = pred.shape[0]
        total_loss = 0
        for b in range(B):
            numb = object_num[b]
            predb = pred[b, :numb, :].unsqueeze(0)
            targetb = target[b, :numb, :].unsqueeze(0)
            lossb = self.bce_loss(predb, targetb.float()) 
            total_loss = total_loss + lossb
        total_loss = total_loss/B
        return total_loss


class BCE_Dice_Loss(nn.Module):

    def __init__(self, weight_bce=1, weight_dc=1, smooth=1.):
        super().__init__()

        self.smooth = smooth
        self.weight_bce = weight_bce
        self.weight_dc = weight_dc
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss(self.smooth)
    
    def forward(self, pred, target):
        ''' pred:(b 1 h w), target:(b h w), object_num:(b) '''
        loss_bce = self.bce_loss(pred.squeeze(1), target.float()) 
        loss_dc = self.dice_loss(pred, target)
        total_loss = self.weight_bce*loss_bce +self.weight_dc*loss_dc
        return total_loss


class Topk_BCE_Dice_Loss(nn.Module):

    def __init__(self, weight_bce=1, weight_dc=1, smooth=1.):
        super().__init__()

        self.smooth = smooth
        self.weight_bce = weight_bce
        self.weight_dc = weight_dc
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss(self.smooth)
    
    def forward(self, pred, target, object_num):
        ''' pred:(b q h w), target:(b q h w), object_num:(b) '''
        B = pred.shape[0]
        total_loss = 0
        for b in range(B):
            numb = object_num[b]
            predb = pred[b, :numb, :, :]
            targetb = target[b, :numb, :, :]
            loss_bce = self.bce_loss(predb, targetb) 
            loss_dc = self.dice_loss(predb, targetb)
            total_loss = total_loss + self.weight_bce*loss_bce +self.weight_dc*loss_dc
        total_loss = total_loss/B
        return total_loss


class Selective_CE_Dice_Loss(nn.Module):
    def __init__(self, weight_bce=1, weight_dc=1, smooth=1e-4):
        super(Selective_CE_Dice_Loss, self).__init__()
        self.weight_bce = weight_bce
        self.weight_dc = weight_dc
        self.smooth = smooth

    def forward(self, preds, targets, object_num): 
        ''' pred:(b q h w), target:(b h w), object_num:(b) '''

        inputs= torch.cat((preds[:, -1:, :, :], preds[:, :-1, :, :]), dim=1)
        B = inputs.shape[0]

        ce_loss = F.cross_entropy(inputs, targets, reduction='none') # b h w
        for b in range(B):
            numb = object_num[b]
            ce_loss[b][targets[b] == 0] = 0 
            ce_loss[b][targets[b] > numb] = 0 
        ce_loss = ce_loss.mean()

        dice_loss = 0
        softmax_inputs = F.softmax(inputs, dim=1) 
        for b in range(B):
            numb = object_num[b]
            dice_loss_b =0
            for i in range(1, numb+1):
                input_i = softmax_inputs[b, i, :, :]
                target_i = (targets[b] == i).float()
                intersection = (input_i * target_i).sum()
                dice_loss_b = dice_loss_b + (1 - (2 * intersection + self.smooth) / (input_i.sum() + target_i.sum() + self.smooth))
            dice_loss = dice_loss + dice_loss_b/numb
        dice_loss = dice_loss / B

        loss = self.weight_bce * ce_loss + self.weight_dc* dice_loss
        return loss




# --------------------------------------------------------------------------------------------------
def custom_cross_entropy_loss(logits, targets, epsilon = 1e-6, minlog=-20):
    probabilities = F.softmax(logits, dim=1)
    log_probabilities = torch.log(probabilities + epsilon)
    #print(torch.max(log_probabilities), torch.min(log_probabilities))
    log_probabilities = torch.clamp(log_probabilities, min=minlog, max=0-epsilon)
    one_hot_targets = F.one_hot(targets, num_classes=logits.shape[1]) # (b h w q)
    one_hot_targets = one_hot_targets.permute(0, 3, 1, 2) # (b q h w)
    loss = -torch.sum(one_hot_targets * log_probabilities, dim=1) # (b h w)
    return loss

class Topk_CE_Dice_Loss(nn.Module):
    def __init__(self, weight_bce=1, weight_dc=1, smooth=1e-4):
        super(Topk_CE_Dice_Loss, self).__init__()
        self.weight_bce = weight_bce
        self.weight_dc = weight_dc
        self.smooth = smooth

    def forward(self, inputs, targets, object_num): 
        ''' pred:(b q h w), target:(b h w), object_num:(b) '''

        B = inputs.shape[0]

        #ce_loss = F.cross_entropy(inputs, targets, reduction='none') # b h w
        # for b in range(B):
        #     numb = object_num[b]
        #     ce_loss[b][targets[b] == 0] = 0 
        #     ce_loss[b][targets[b] > numb] = 0 
        # ce_loss = ce_loss.mean()

        ce_loss = custom_cross_entropy_loss(inputs, targets)
        valid = 0
        for b in range(B):
            numb = object_num[b]
            valid = valid + torch.sum(targets[b] > 0)
            valid = valid - torch.sum(targets[b] > numb)
            ce_loss[b][targets[b] == 0] = 0 
            ce_loss[b][targets[b] > numb] = 0 
        ce_loss = torch.sum(ce_loss) / (valid +1)

        # --------------------------------------------------

        dice_loss = 0
        softmax_inputs = F.softmax(inputs, dim=1) 
        for b in range(B):
            numb = object_num[b]
            dice_loss_b =0
            for i in range(1, numb+1):
                input_i = softmax_inputs[b, i, :, :]
                target_i = (targets[b] == i).float()
                intersection = (input_i * target_i).sum()
                dice_loss_b = dice_loss_b + (1 - (2 * intersection + self.smooth) / (input_i.sum() + target_i.sum() + self.smooth))
            dice_loss = dice_loss + dice_loss_b/numb
        dice_loss = dice_loss / B

        loss = self.weight_bce * ce_loss + self.weight_dc* dice_loss
        return  loss


class CE_Dice_Loss(nn.Module):
    def __init__(self, weight_bce=0.5, weight_dc=0.5, smooth=1e-4):
        super(CE_Dice_Loss, self).__init__()
        self.weight_bce = weight_bce
        self.weight_dc = weight_dc
        self.smooth = smooth

    def forward(self, inputs, targets): 
        ''' pred:(b 2 h w), target:(b h w), object_num:(b) '''

        B = inputs.shape[0]
        ce_loss = F.cross_entropy(inputs, targets.long(), reduction='none') # b h w
        ce_loss = ce_loss.mean()

        dice_loss = 0
        softmax_inputs = F.softmax(inputs, dim=1) 
        for b in range(B):
            dice_loss_b =0
            for i in range(inputs.shape[1]):
                input_i = softmax_inputs[b, i, :, :]
                target_i = (targets[b] == i).float()
                intersection = (input_i * target_i).sum()
                dice_loss_b = dice_loss_b + (1 - (2 * intersection + self.smooth) / (input_i.sum() + target_i.sum() + self.smooth))
            dice_loss = dice_loss + dice_loss_b/inputs.shape[1]
        dice_loss = dice_loss / B

        loss = self.weight_bce * ce_loss + self.weight_dc* dice_loss
        return loss
    



# =================================================================================================================================================
class Orderk_BCE_Dice_Loss(nn.Module):

    def __init__(self, weight_bce=1, weight_dc=1, smooth=1.):
        super().__init__()

        self.smooth = smooth
        self.weight_bce = weight_bce
        self.weight_dc = weight_dc
    
    def forward(self, pred, target, object_num, merged_gt):
        ''' pred:(b q h w), target:(b q h w), object_num:(b), merged_gt: (b h w) '''
        B = pred.shape[0]

        total_loss = 0
        for b in range(B):
            numb = object_num[b]
            predb = pred[b, :numb, :, :]
            targetb = target[b, :numb, :, :]
            mergedb = merged_gt[b, :, :].unsqueeze(0).repeat(predb.shape[0], 1, 1) # (k h w)

            predbsig = torch.sigmoid(predb) # (k h w)
            predbsig = predbsig.contiguous().view(numb, -1) # (k N)
            targetbsig = targetb.contiguous().view(numb, -1) # (k N)
            mergedb = mergedb.contiguous().view(numb, -1)
            intersection = torch.sum(predbsig * targetbsig, dim=-1) # (k)
            union = torch.sum(predbsig, dim=-1) + torch.sum(targetbsig, dim=-1)
            dice = (2 * intersection + self.smooth) / (union + self.smooth) # (k)
            loss_global_dc = (1 - dice).mean()

            # Foreground1: I donn't have, but merged have
            loss_foreground1 = -torch.log(1- predbsig + 1e-4)*(1 - targetbsig)*mergedb
            loss_foreground1 = torch.sum(loss_foreground1, dim=-1) / (torch.sum((1 - targetbsig)*mergedb, dim=-1) + 1)
            loss_foreground1 = loss_foreground1.mean()

            # Foreground2: I donn't have, and merged donn't have
            loss_foreground2 = -torch.log(1- predbsig + 1e-4)*(1 - targetbsig)*(1 - mergedb)
            loss_foreground2 = torch.sum(loss_foreground2, dim=-1) / (torch.sum((1 - targetbsig)*(1 - mergedb), dim=-1) + 1)
            loss_foreground2 = loss_foreground2.mean()

            # Foreground3: I have
            nump = torch.sum(predbsig*targetbsig > 0.5, dim=-1)
            numg = torch.sum(targetbsig > 0, dim=-1)
            loss_foreground3 = -torch.log(predbsig + 1e-4)*targetbsig
            loss_foreground3 = torch.sum(loss_foreground3, dim=-1) / (torch.sum(targetbsig, dim=-1) + 1)
            loss_foreground3 = loss_foreground3.mean()

            #loss_foreground = loss_foreground1 + loss_foreground2 + 1*loss_foreground3
            loss_foreground = torch.max(loss_foreground1, loss_foreground3)
            loss_foreground =  torch.max(loss_foreground, loss_foreground2)
           
            predb2 = pred[b, numb:, :, :]
            predb2 = predb2.contiguous().view(predb2.shape[0], -1)
            mergedb2 = merged_gt[b, :, :].unsqueeze(0).repeat(predb2.shape[0], 1, 1) # (m h w)
            mergedb2 = mergedb2.contiguous().view(predb2.shape[0], -1) # (m -1)
            predb2sig = torch.sigmoid(predb2) #(m -1)
            loss_background = -torch.log(1 - predb2sig + 1e-4) * mergedb2
            loss_background = torch.sum(loss_background, dim=-1) /  (torch.sum(mergedb2, dim=-1) + 1)
            loss_background = loss_background.mean()

            #total_loss = total_loss + self.weight_bce*(loss_foreground + loss_background) + self.weight_dc*loss_global_dc 
            total_loss = total_loss + 0.5*loss_foreground +0.2*loss_background + 0.3*loss_global_dc

        return total_loss / B


class OrderMatching(nn.Module):
    def __init__(self, penalty_margin=0.001, margin=0.001):
        super().__init__()
        self.penalty_margin = penalty_margin
        self.margin = margin # 0.001

    def forward(self, query, text, text_labels, object_num):
        """
        :param query: (b, q, d) 
        :param text: (b, l, d)
        :param text_labels: (b, l)
        :param object_num: (b)
        """
        batch_size, q, d = query.shape
        _, l, _ = text.shape
        text_labels[text_labels==-1] = q

        # Similarity scores of query and text
        similarity_matrix = torch.matmul(query, text.transpose(1, 2))  # (b, q, l)
        similarity_matrix = similarity_matrix * (d ** -0.5) # b q l
        similarity_matrix = F.softmax(similarity_matrix, dim=1)

        total_loss = 0
        for b in range(batch_size):
            contrastive_loss = 0
            for t_idx in range(l):
                gt_object = text_labels[b, t_idx]
                object_query_idx = gt_object-1
                sub_loss =  1 - similarity_matrix[b, object_query_idx, t_idx]
                contrastive_loss = contrastive_loss + sub_loss
            contrastive_loss =  contrastive_loss / l

            total_loss = total_loss + contrastive_loss
        
        # ------ orthogonality_loss -----------------
        orthogonality_loss = 0
        size_loss = 0
        for qi  in range(q):
            for qj in range(qi+1, q):
                oloss_ij = torch.sum(query[:, qi, :]*query[:, qj, :], dim=1)
                oloss_ij = torch.abs(oloss_ij)
                orthogonality_loss = orthogonality_loss + oloss_ij
            size = torch.sum(query[:, qi, :]**2, dim=1)
            size = torch.clamp(1-size, min=0)
            size_loss = size_loss + size

        orthogonality_loss =  orthogonality_loss/(q*(q-1)*0.5)
        orthogonality_loss = torch.mean(orthogonality_loss)
        size_loss =  size_loss/qi
        size_loss = torch.mean(size_loss)

        return total_loss / batch_size + (0.5*size_loss+0.5*orthogonality_loss)


class DynamicMasksLoss_w_Matching(nn.Module):

    def __init__(self, num_query=25, w_merged=0.2, w_masks=0.6, w_class=0.1, w_num=0.1, w_bce=1, w_dc=1, w_om=0.05, smooth=1.):
        super().__init__()

        self.num_query = num_query
        self.smooth = smooth
        self.w_merged = w_merged
        self.w_masks = w_masks
        self.w_class = w_class
        self.w_num = w_num
        self.w_om = w_om

        self.topk_bce_dice_loss = Orderk_BCE_Dice_Loss(w_bce, w_dc, self.smooth)

        self.bce_dice_loss = CE_Dice_Loss(w_bce, w_dc, self.smooth)

        self.topk_ce_loss = TopkCELoss()

        self.matching_loss = OrderMatching()

        self.smooth_l1 = nn.SmoothL1Loss()
    
    def forward(self, merged_pred, merged_gt, masks_pred, masks_gt, cquery_pred, cquery_gt, obnum_pred, obnum_gt, query, text, text_label):
        ''' 
        merged_pred:(b 1 h w), masks_pred:(b q h w), cquery_pred:(b, q, 2), obnum_pred:(b) 
        merged_gt:(b h w),   masks_gt:(b q h w),   cquery_gt:(b q),       obnum_gt:(b)
        '''
        loss_merged = self.bce_dice_loss(merged_pred, merged_gt)
        loss_masks = self.topk_bce_dice_loss(masks_pred, masks_gt, obnum_gt, merged_gt)
        loss_class = self.topk_ce_loss(cquery_pred, cquery_gt, obnum_gt)
        loss_num = self.smooth_l1(obnum_pred, obnum_gt.float()/self.num_query)
        loss_matching = self.matching_loss(query, text, text_label, obnum_gt)

        total_loss = loss_merged * self.w_merged + loss_masks * self.w_masks + loss_class * self.w_class + loss_num * self.w_num + loss_matching * self.w_om

        return {"total_loss": total_loss, "loss_merged": loss_merged, "loss_masks": loss_masks, "loss_class": loss_class, "loss_num": loss_num, "loss_matching": loss_matching}

