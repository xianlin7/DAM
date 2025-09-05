# this file is utilized to evaluate the models from different mode: 2D-slice level, 2D-patient level, 3D-patient level
from torch.autograd import Variable
from torch.utils.data import DataLoader
import os
import numpy as np
import torch
import torch.nn.functional as F
import utils.metrics as metrics
from hausdorff import hausdorff_distance
from utils.visualization import visual_segmentation
from utils.visualization_dam import visual_segmentation_single, visual_image_heatmap, visual_segmentation_any, visual_segmentation_order
import pandas as pd


def eval_slice(valloader, model, opt):
    model.eval()
    val_losses, mean_dice = 0, 0
    dices, hds = 0, 0
    ious, accs, ses, sps = 0, 0, 0, 0
    eval_number = 0
    dices_no, dices_multi, iou_no, iou_multi = 0, 0, 0, 0
    num_no, num_multi = 0, 0
    for batch_idx, (datapack) in enumerate(valloader):

        with torch.no_grad():
            model_output = model(datapack)

        val_loss = model_output['loss']
        val_losses += val_loss.item()

        gt_mask = Variable(datapack['mask_merged'].to(device=opt.device))
        gt_mask = gt_mask.detach().cpu().numpy()

        if model_output['predict'].shape[1] == 2:
            predict = F.softmax(model_output['predict'], dim=1)
            predict = predict.detach().cpu().numpy()  # (b, c, h, w)
            seg = np.argmax(predict, axis=1)  # (b, h, w)
        else:
            predict = torch.sigmoid(model_output['predict'])
            predict = predict.detach().cpu().numpy() # (b, 1, h, w)
            seg = predict[:, 0, :, :] > 0.5  # (b, h, w)
        
        if model_output["existing"] is not None:
            existing = torch.sigmoid(model_output["existing"])
            existing = existing.detach().cpu().numpy()
            existing = existing > 0.5

        b, h, w = seg.shape
        for j in range(0, b):
            pred_j = np.zeros((1, h, w))
            pred_j[seg[j:j+1, :, :] == 1] = 255
            if model_output["existing"] is not None and existing[j] == 0:
                pred_j = pred_j*0
            gt_j = np.zeros((1, h, w))
            gt_j[gt_mask[j:j+1, :, :] == 1] = 255
            #dices += metrics.dice_coefficient(pred_j, gt_j)
            dice = metrics.dice_coefficient(pred_j, gt_j)
            iou, acc, se, sp = metrics.sespiou_coefficient(pred_j, gt_j, all=False)
            dices += dice
            ious += iou
            accs += acc
            ses += se
            sps += sp
            hds += hausdorff_distance(pred_j[0, :, :], gt_j[0, :, :], distance="manhattan")

            if not datapack['referring_existing'][j]:
                num_no = num_no + 1
                dices_no = dices_no + dice
                iou_no = iou_no + iou
            
            if datapack['class_num'][j] > 1:
                num_multi = num_multi+1
                dices_multi = dices_multi + dice
                iou_multi = iou_multi + iou

            del pred_j, gt_j
            if opt.visual:
                visual_segmentation(seg[j:j+1, :, :], datapack['image_name'][j], opt)
        eval_number = eval_number + b
    mean_dice = dices / eval_number
    mean_hdis = hds / eval_number
    mean_iou, mean_acc, mean_se, mean_sp = ious/eval_number, accs/eval_number, ses/eval_number, sps/eval_number
    val_losses = val_losses / (batch_idx + 1)

    # mean_dice_no = dices_no/num_no
    # mean_iou_no = iou_no/ num_no
    # mean_dice_multi = dices_multi/num_multi
    # mean_iou_multi = iou_multi/num_multi
    # print("num no:", num_no, "  num multi:", num_multi)
    # print("dice_no:", mean_dice_no, "iou_no;", mean_iou_no, "dice_multi:", mean_dice_multi, "iou_multi:", mean_iou_multi)
    
    return {"mdice":mean_dice, "loss": val_losses, "mhd":mean_hdis, "miou":mean_iou, "macc":mean_acc, "mse": mean_se, "msp": mean_sp}

def eval_slice_record(valloader, model, opt):
    model.eval()
    val_losses, mean_dice = 0, 0
    dices, hds = 0, 0
    ious, accs, ses, sps = 0, 0, 0, 0
    eval_number = 0
    dices_no, dices_multi, iou_no, iou_multi = 0, 0, 0, 0
    num_no, num_multi = 0, 0

    keep_excel = []
   
    for batch_idx, (datapack) in enumerate(valloader):

        with torch.no_grad():
            model_output = model(datapack)

        val_loss = model_output['loss']
        val_losses += val_loss.item()

        gt_mask = Variable(datapack['mask_merged'].to(device=opt.device))
        gt_mask = gt_mask.detach().cpu().numpy()

        if model_output['predict'].shape[1] == 2:
            predict = F.softmax(model_output['predict'], dim=1)
            predict = predict.detach().cpu().numpy()  # (b, c, h, w)
            seg = np.argmax(predict, axis=1)  # (b, h, w)
        else:
            predict = torch.sigmoid(model_output['predict'])
            predict = predict.detach().cpu().numpy() # (b, 1, h, w)
            seg = predict[:, 0, :, :] > 0.5  # (b, h, w)
        
        if model_output["existing"] is not None:
            existing = torch.sigmoid(model_output["existing"])
            existing = existing.detach().cpu().numpy()
            existing = existing > 0.5

        b, h, w = seg.shape
        for j in range(0, b):
            pred_j = np.zeros((1, h, w))
            pred_j[seg[j:j+1, :, :] == 1] = 255
            if model_output["existing"] is not None and existing[j] == 0:
                pred_j = pred_j*0
            gt_j = np.zeros((1, h, w))
            gt_j[gt_mask[j:j+1, :, :] == 1] = 255
            #dices += metrics.dice_coefficient(pred_j, gt_j)
            dice = metrics.dice_coefficient(pred_j, gt_j)
            iou, acc, se, sp = metrics.sespiou_coefficient(pred_j, gt_j, all=False)
            dices += dice
            ious += iou
            accs += acc
            ses += se
            sps += sp
            hds += hausdorff_distance(pred_j[0, :, :], gt_j[0, :, :], distance="manhattan")

            visual_segmentation_single(pred_j, datapack['image_path'][j], datapack['image_name'][j], datapack['class_index'][j], str(0))


            if not datapack['referring_existing'][j]:
                num_no = num_no + 1
                dices_no = dices_no + dice
                iou_no = iou_no + iou           

            if datapack['class_num'][j] > 1:
                num_multi = num_multi+1
                dices_multi = dices_multi + dice
                iou_multi = iou_multi + iou
            
            new_row = {}
            new_row["Dataset Name"] = datapack['data_name'][j]
            new_row["Class Index"] =  datapack['class_index'][j]
            new_row["Image Name"] =  datapack['image_name'][j]
            new_row["Description"] =  datapack['referring_txt'][j]
            new_row["Text Label"] =  "1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1"
            new_row["RecLMIS"] = dice
            keep_excel.append(new_row)

            del pred_j, gt_j
            if opt.visual:
                visual_segmentation(seg[j:j+1, :, :], datapack['image_name'][j], opt)
        eval_number = eval_number + b
    mean_dice = dices / eval_number
    mean_hdis = hds / eval_number
    mean_iou, mean_acc, mean_se, mean_sp = ious/eval_number, accs/eval_number, ses/eval_number, sps/eval_number
    val_losses = val_losses / (batch_idx + 1)

    keep_excel = pd.DataFrame(keep_excel)
    keep_excel.to_excel('RecLMIS.xlsx', index=False)
    
    return {"mdice":mean_dice, "loss": val_losses, "mhd":mean_hdis, "miou":mean_iou, "macc":mean_acc, "mse": mean_se, "msp": mean_sp}


def eval_dm_slice(valloader, model, opt):
    model.eval()
    val_losses = 0
    dices, ious, ldices, lious = 0, 0, 0, 0
    merged_dices, merged_ious, merged_ldices, merged_lious = 0, 0, 0, 0
    oTPs, oFPs, oFNs = 0, 0, 0
    eval_number = 0
    for batch_idx, (datapack) in enumerate(valloader):

        with torch.no_grad():
            model_output = model(datapack)

        val_loss = model_output['loss']
        val_losses += val_loss.item()

        masks_gt = datapack['mask_instances'].to(dtype = torch.float32, device=opt.device) # (b q h w)
        merged_gt = datapack['mask_merged'].to(dtype = torch.float32, device=opt.device) # (b h w)
        cquery_gt = datapack['mask_existing'].to(dtype = torch.long, device=opt.device) # (b q)
        obnum_gt = datapack['class_num'].to(dtype = torch.long, device=opt.device) # (b)

        masks_gt = masks_gt.detach().cpu().numpy() # (b q h w)
        obnum_gt = obnum_gt.detach().cpu().numpy()
        existing_gt = cquery_gt.detach().cpu().numpy() # (b q)

        pred_existing = F.softmax(model_output['pred_cquery'], dim=-1) # b q 2
        pred_existing = pred_existing.detach().cpu().numpy() # (b, q, 2)
        pred_existing = np.argmax(pred_existing, axis=-1) # (b, q)

        pred_num = model_output['pred_num'].detach().cpu().numpy()

        pred_masks = torch.sigmoid(model_output['pred_masks']) # (b q h w)
        pred_masks = pred_masks.detach().cpu().numpy() # (b q h w)
        seg = pred_masks > 0.5  # (b q h w)

        b, q, h, w = seg.shape
        for i in range(0, b):
            pred_numi = round(pred_num[i])
            gt_numi = obnum_gt[i] # (1)
            pred_existingi = pred_existing[i][:gt_numi] # (q)
            gt_existingi = existing_gt[i][:gt_numi]
            segi = seg[i, :gt_numi, :, :] # (q h w)
            masksi = masks_gt[i, :gt_numi, :, :]

            pred_i = np.zeros((gt_numi, h, w))    
            pred_i[segi[pred_existingi>0, :, :] == 1] = 255
            gt_i = np.zeros((gt_numi, h, w))
            gt_i[masksi == 1] = 255

            dicei, ioui, ldicei, lioui, otp, ofp, ofn = metrics.mm_coefficient(pred_i, gt_i, pred_numi, gt_numi, gt_existingi, thresh=0.5)
            dices, ious, ldices, lious = dices+dicei, ious+ioui, ldices+ldicei, lious+lioui
            oTPs, oFPs, oFNs = oTPs+otp, oFPs+ofp, oFNs+ofn
               
            if opt.visual:
                text_refering = datapack['referring_txt'][i]
                class_index = datapack['class_index'][i]
                visual_segmentation(segi, datapack['image_path'][i], opt)
        
        # ------ compute merged value for comparison -----------
        merged_pred = np.zeros((b, h, w))
        reference_existing = datapack['referring_existing'].to(dtype = torch.long, device=opt.device).detach().cpu().numpy()
        merged_gt = merged_gt.detach().cpu().numpy() # (b h w)
        for i in range(0, b):
            pred_numi = round(pred_num[i])
            pred_existingi = pred_existing[i][:pred_numi] # (k)
            segi = seg[i, :pred_numi, :, :] # (k h w)
            segi[pred_existingi==0, :, :] = 0
            segi = np.sum(segi, axis=0) #  (h w)
            segi[segi>=1] = 1
            merged_pred[i, :, :] = segi

        predict = torch.sigmoid(model_output['predict']).squeeze(1)
        predict = predict.detach().cpu().numpy() # (b, h, w)
        merged_pred2 = predict[:, :, :] > 0.5  # (b, h, w)

        merged_dice, merged_iou, merged_ldice, merged_liou = metrics.merged_coefficient(merged_pred2, merged_gt, pred_num)
        merged_dices, merged_ious, merged_ldices, merged_lious = merged_dices + merged_dice, merged_ious + merged_iou, merged_ldices+merged_ldice, merged_lious+merged_liou

        eval_number = eval_number + b

    mean_dice, mean_iou, mean_ldice, mean_liou  = dices / eval_number, ious / eval_number, ldices / eval_number, lious / eval_number
    mean_merged_dice, mean_merged_iou, mean_merged_ldice, mean_merged_liou = merged_dices/eval_number, merged_ious/eval_number, merged_ldices/eval_number, merged_lious/eval_number
    Pre = (oTPs + 1e-5) / (oTPs + oFPs + 1e-5)
    Recall = (oTPs + 1e-5) / (oTPs + oFNs + 1e-5)
    val_losses = val_losses / (batch_idx + 1)
    
    return {"mdice":mean_dice, "loss": val_losses, "miou":mean_iou, "mldice":mean_ldice, "mliou": mean_liou, "Pre":Pre, "Recall":Recall,
            "merged_mdices":mean_merged_dice, "merged_miou":mean_merged_iou, "merged_mldice":mean_merged_ldice, "merged_liou":mean_merged_liou}


def eval_merged_slice(valloader, model, opt):
    model.eval()
    val_losses = 0
    merged_dices, merged_ious, merged_ldices, merged_lious = 0, 0, 0, 0
    eval_number = 0
    for batch_idx, (datapack) in enumerate(valloader):

        with torch.no_grad():
            model_output = model(datapack)

        val_loss = model_output['loss']
        val_losses += val_loss.item()

        merged_gt = datapack['mask_merged'].to(dtype = torch.float32, device=opt.device) # (b h w)
        merged_gt = merged_gt.detach().cpu().numpy() # (b h w)
        reference_existing = datapack['referring_existing'].to(dtype = torch.long, device=opt.device).detach().cpu().numpy() #(b)

        pred_existing = F.softmax(model_output['pred_cquery'], dim=-1) # b q 2
        pred_existing = pred_existing.detach().cpu().numpy() # (b, q, 2)
        pred_existing = np.argmax(pred_existing, axis=-1) # (b, q)

        pred_num = model_output['pred_num'].detach().cpu().numpy()

        pred_masks = torch.sigmoid(model_output['pred_masks']) # (b q h w)
        pred_masks = pred_masks.detach().cpu().numpy() # (b q h w)
        seg = pred_masks > 0.5  # (b q h w)

        b, q, h, w = seg.shape
        merged_pred = np.zeros((b, h, w))
        for i in range(0, b):
            pred_numi = round(pred_num[i])
            pred_existingi = pred_existing[i][:pred_numi] # (k)
            #print(pred_existingi)
            segi = seg[i, :pred_numi, :, :] # (k h w)
            #print("num", pred_numi)
            segi[pred_existingi==0, :, :] = 0
            segi = np.sum(segi, axis=0) #  (h w)
            segi[segi>=1] = 1
            merged_pred[i, :, :] = segi

            if opt.visual:
                text_refering = datapack['referring_txt'][i]
                class_index = datapack['class_index'][i]
                visual_segmentation(segi, datapack['image_path'][i], opt)
        
        predict = torch.sigmoid(model_output['predict']).squeeze(1)
        predict = predict.detach().cpu().numpy() # (b, h, w)
        merged_pred2 = predict[:, :, :] > 0.5  # (b, h, w)
        #print(np.sum(merged_pred), "  ", np.sum(merged_gt))
        merged_dice, merged_iou, merged_ldice, merged_liou = metrics.merged_coefficient(merged_pred, merged_gt, reference_existing, pred_num)
        
        merged_dices, merged_ious, merged_ldices, merged_lious = merged_dices + merged_dice, merged_ious + merged_iou, merged_ldices+merged_ldice, merged_lious+merged_liou

        eval_number = eval_number + b

    mean_merged_dice, mean_merged_iou, mean_merged_ldice, mean_merged_liou = merged_dices/eval_number, merged_ious/eval_number, merged_ldices/eval_number, merged_lious/eval_number
    val_losses = val_losses / (batch_idx + 1)
    
    return {"mdice":mean_merged_dice, "loss": val_losses, "miou":mean_merged_iou, "mldice":mean_merged_ldice, "mliou": mean_merged_liou,
            "merged_mdices":mean_merged_dice, "merged_miou":mean_merged_iou, "merged_mldice":mean_merged_ldice, "merged_liou":mean_merged_liou}


def eval_combined_slice(valloader, model, opt):
    model.eval()
    val_losses = 0
    merged_dices, merged_ious, merged_ldices, merged_lious = 0, 0, 0, 0
    combined_dices, combined_ious, combined_ldices, combined_lious = 0, 0, 0, 0
    eval_number = 0
    for batch_idx, (datapack) in enumerate(valloader):

        with torch.no_grad():
            model_output = model(datapack)

        val_loss = model_output['loss']
        val_losses += val_loss.item()

        merged_gt = datapack['mask_merged'].to(dtype = torch.float32, device=opt.device) # (b h w)
        merged_gt = merged_gt.detach().cpu().numpy() # (b h w)
        combined_gt = datapack['mask_combined'].to(dtype = torch.float32, device=opt.device) # (b h w)
        combined_gt = combined_gt.detach().cpu().numpy() # (b h w)
        reference_existing = datapack['referring_existing'].to(dtype = torch.long, device=opt.device).detach().cpu().numpy() #(b)
        obnum_gt = datapack['class_num'].to(dtype = torch.long, device=opt.device).detach().cpu().numpy() #(b)

        pred_existing = F.softmax(model_output['pred_cquery'], dim=-1) # b q 2
        pred_existing = pred_existing.detach().cpu().numpy() # (b, q, 2)
        pred_existing = np.argmax(pred_existing, axis=-1) # (b, q)

        pred_num = model_output['pred_num'].detach().cpu().numpy()

        pred_masks = model_output['pred_masks']  # (b q h w)
        pred_masks = torch.cat((pred_masks[:, -1:, :, :], pred_masks[:, :-1, :, :]), dim=1)
        pred_masks =  F.softmax(pred_masks, dim=1) # (b q h w)
        pred_masks = pred_masks.detach().cpu().numpy() # (b q h w)
        seg = np.argmax(pred_masks, axis=1)  # (b, h, w)

        b, h, w = seg.shape
        merged_pred = np.zeros((b, h, w))
        for i in range(0, b):
            pred_numi = round(pred_num[i])
            gt_numi = obnum_gt[i]
            combined_dice, combined_iou, combined_ldice, combined_liou = 0, 0, 0, 0
            for c in range(1, gt_numi+1):
                pred_c = np.zeros((1, h, w))
                if pred_existing[i, c-1] > 0:
                    pred_c[seg[i:i+1, :, :] == c] = 1
                gt_c = np.zeros((1, h, w))
                gt_c[combined_gt[i:i+1, :, :] == c] = 1
                dice_c, iou_c, ldice_c, liou_c = metrics.combined_coefficient(pred_c, gt_c, reference_existing[i:i+1], pred_existing[i, c-1])
                combined_dice, combined_iou, combined_ldice, combined_liou =  combined_dice + dice_c, combined_iou + iou_c, combined_ldice + ldice_c, combined_liou +liou_c
            combined_dice, combined_iou, combined_ldice, combined_liou = combined_dice/gt_numi, combined_iou/gt_numi, combined_ldice/gt_numi, combined_liou/gt_numi
            combined_dices, combined_ious, combined_ldices, combined_lious =  combined_dices + combined_dice, combined_ious + combined_iou, combined_ldices + combined_ldice, combined_lious + combined_liou

            pred_existingi = pred_existing[i][:pred_numi] # (k)
            segi = seg[i, : , :] # (h w)
            segi[segi > pred_numi] = 0 
            for c in range(pred_numi):
                if pred_existingi[c] == 0:
                    segi[segi == c+1] = 0
            segi[segi > 0] = 1
            merged_pred[i, :, :] = segi

            if opt.visual:
                text_refering = datapack['referring_txt'][i]
                class_index = datapack['class_index'][i]
                visual_segmentation(segi, datapack['image_path'][i], opt)
        
        predict = torch.sigmoid(model_output['predict']).squeeze(1)
        predict = predict.detach().cpu().numpy() # (b, h, w)
        merged_pred2 = predict[:, :, :] > 0.5  # (b, h, w)
        #print(np.sum(merged_pred), "  ", np.sum(merged_gt))

        merged_dice, merged_iou, merged_ldice, merged_liou = metrics.merged_coefficient(merged_pred, merged_gt, reference_existing, pred_num)
        
        merged_dices, merged_ious, merged_ldices, merged_lious = merged_dices + merged_dice, merged_ious + merged_iou, merged_ldices+merged_ldice, merged_lious+merged_liou

        eval_number = eval_number + b

    mean_merged_dice, mean_merged_iou, mean_merged_ldice, mean_merged_liou = merged_dices/eval_number, merged_ious/eval_number, merged_ldices/eval_number, merged_lious/eval_number
    
    mean_combined_dice, mean_combined_iou, mean_combined_ldice, mean_combined_liou = combined_dices/eval_number, combined_ious/eval_number, combined_ldices/eval_number, combined_lious/eval_number
    
    val_losses = val_losses / (batch_idx + 1)
    
    return {"mdice":mean_merged_dice, "loss": val_losses, "miou":mean_merged_iou, "mldice":mean_merged_ldice, "mliou": mean_merged_liou,
            "merged_mdices":mean_combined_dice, "merged_miou":mean_combined_iou, "merged_mldice":mean_combined_ldice, "merged_liou":mean_combined_liou}


def eval_dm_patient(valloader, model, opt):
    model.eval()
    val_losses = 0
    RecordDict = {}

    for batch_idx, (datapack) in enumerate(valloader):

        with torch.no_grad():
            model_output = model(datapack)

        val_loss = model_output['loss']
        val_losses += val_loss.item()

        masks_gt = datapack['mask_instances'].to(dtype = torch.float32, device=opt.device) # (b q h w)
        merged_gt = datapack['mask_merged'].to(dtype = torch.float32, device=opt.device) # (b h w)
        cquery_gt = datapack['mask_existing'].to(dtype = torch.long, device=opt.device) # (b q)
        obnum_gt = datapack['class_num'].to(dtype = torch.long, device=opt.device) # (b)
        reference_existing = datapack['referring_existing'].to(dtype = torch.long, device=opt.device).detach().cpu().numpy()

        merged_gt = merged_gt.detach().cpu().numpy() # (b h w)
        masks_gt = masks_gt.detach().cpu().numpy() # (b q h w)
        obnum_gt = obnum_gt.detach().cpu().numpy()
        existing_gt = cquery_gt.detach().cpu().numpy() # (b q)

        pred_existing = F.softmax(model_output['pred_cquery'], dim=-1) # b q 2
        pred_existing = pred_existing.detach().cpu().numpy() # (b, q, 2)
        pred_existing = np.argmax(pred_existing, axis=-1) # (b, q)

        pred_num = model_output['pred_num'].detach().cpu().numpy()

        pred_masks = torch.sigmoid(model_output['pred_masks']) # (b q h w)
        pred_masks = pred_masks.detach().cpu().numpy() # (b q h w)
        seg = pred_masks > 0.5  # (b q h w)

        predict_merged = torch.sigmoid(model_output['predict']).squeeze(1)
        predict_merged = predict_merged.detach().cpu().numpy() # (b, h, w)
        predict_merged = predict_merged > 0.5


        b, q, h, w = seg.shape
        for i in range(0, b):
            patient_key = datapack['patient_id'][i] + "-" + datapack['referring_txt'][i]
            class_index = datapack['class_index'][i]
            pred_numi = round(pred_num[i])
            pred_existingi = pred_existing[i] # (q)

            Patient_dict = RecordDict.get(patient_key, {})

            classes = str(class_index).split("-")
            for cth, (c) in enumerate(classes):
    
                patient_c_dict = Patient_dict.get(c, {})

                pred_j = np.zeros((1, h, w))
                gt_j = np.zeros((1, h, w))
                gt_j[masks_gt[i, cth:cth+1, :, :]==1] =255
                if cth < pred_numi:
                    if pred_existingi[cth] == 1:
                        pred_j[seg[i, cth:cth+1, :, :]==1] = 255
                    tp, fp, tn, fn = metrics.get_matrix(pred_j, gt_j)
                else:
                    # haven't predict this mask
                    tp, fp, tn, fn = 0, 0, 0, h*w
                gtexist = existing_gt[i, cth]
                
                patient_c_dict["tp"] = patient_c_dict.get("tp", 0) + tp
                patient_c_dict["fp"] = patient_c_dict.get("fp", 0) + fp
                patient_c_dict["tn"] = patient_c_dict.get("tn", 0) + tn
                patient_c_dict["fn"] = patient_c_dict.get("fn", 0) + fn
                patient_c_dict["gtexist"] = patient_c_dict.get("gtexist", 0) + gtexist

                Patient_dict[c] = patient_c_dict
            
            # ------ compute merged value for comparison ----------
            patient_merged_dict = Patient_dict.get("merged", {})
            merged_pred = np.zeros((1, h, w))
            if pred_numi > 0:
                pred_existingi = pred_existingi[:pred_numi] # (k)
                segi = seg[i, :pred_numi, :, :] # (k h w)
                segi[pred_existingi==0, :, :] = 0
                segi = np.sum(segi, axis=0) #  (h w)
                segi[segi>=1] = 1
                merged_pred[0, :, :] = segi

                merged_pred2 = predict_merged[i:i+1, :, :]
                tp, fp, tn, fn = metrics.get_matrix(merged_pred, merged_gt[i:i+1, :, :])
            else:
                tp, fp, tn, fn = 0, 0, 0, h*w

            patient_merged_dict["tp"] = patient_merged_dict.get("tp", 0) + tp
            patient_merged_dict["fp"] = patient_merged_dict.get("fp", 0) + fp
            patient_merged_dict["tn"] = patient_merged_dict.get("tn", 0) + tn
            patient_merged_dict["fn"] = patient_merged_dict.get("fn", 0) + fn
            patient_merged_dict["gtexist"] = patient_merged_dict.get("gtexist", 0) + reference_existing[i]

            Patient_dict["merged"] = patient_merged_dict

            RecordDict[patient_key] = Patient_dict

            if opt.visual:
                visual_segmentation(seg[i, :, :, :], datapack['image_path'][i], opt)
    
    val_losses = val_losses / (batch_idx + 1)

    # process the Record Dict to compute the metrics
    smooth = 1e-5
    DicesD, IousD, LDicesD, LIousD = {}, {}, {}, {}
    Merged_DicesD, Merged_IousD, Merged_LDicesD, Merged_LIousD = {}, {}, {}, {}
    for key in RecordDict:
        Patient_dict = RecordDict[key]
        dices, ious, ldices, lious = [], [], [], []
        for classkey in Patient_dict:
            cdict = Patient_dict[classkey]
            if classkey != "merged":
                dices.append((2 * cdict["tp"] + smooth) / (2 * cdict["tp"] + cdict["fp"] + cdict["fn"] + smooth))
                ious.append((cdict["tp"] + smooth) / (cdict["tp"] + cdict["fp"] + cdict["fn"] + smooth))
                if cdict["gtexist"] > 0:
                    ldices.append((2 * cdict["tp"] + smooth) / (2 * cdict["tp"] + cdict["fp"] + cdict["fn"] + smooth))
                    lious.append((cdict["tp"] + smooth) / (cdict["tp"] + cdict["fp"] + cdict["fn"] + smooth))
                else:
                    ldices.append(1 / (cdict["fp"] + cdict["fn"] + 1))
                    lious.append(1 / (cdict["fp"] + cdict["fn"] + 1))
            else:
                Merged_DicesD[key] = (2 * cdict["tp"] + smooth) / (2 * cdict["tp"] + cdict["fp"] + cdict["fn"] + smooth)
                Merged_IousD[key] = (cdict["tp"] + smooth) / (cdict["tp"] + cdict["fp"] + cdict["fn"] + smooth)
                if cdict["gtexist"] > 0:
                    Merged_LDicesD[key] = (2 * cdict["tp"] + smooth) / (2 * cdict["tp"] + cdict["fp"] + cdict["fn"] + smooth)
                    Merged_LIousD[key] = (cdict["tp"] + smooth) / (cdict["tp"] + cdict["fp"] + cdict["fn"] + smooth)
                else:
                    Merged_LDicesD[key] = 1 / (cdict["fp"] + cdict["fn"] + 1)
                    Merged_LIousD[key] = 1 / (cdict["fp"] + cdict["fn"] + 1)

        DicesD[key], IousD[key], LDicesD[key], LIousD[key] = sum(dices) / len(dices), sum(ious) / len(ious), sum(ldices) / len(ldices), sum(lious) / len(lious)
    
    DicesD, Merged_DicesD, IousD, Merged_IousD  = list(DicesD.values()), list(Merged_DicesD.values()), list(IousD.values()), list(Merged_IousD.values())
    LDicesD, Merged_LDicesD, LIousD, Merged_LIousD = list(LDicesD.values()), list(Merged_LDicesD.values()), list(LIousD.values()), list(Merged_LIousD.values())

    Dices_ = np.array([x.item() if isinstance(x, np.ndarray) else x for x in DicesD])
    Merged_Dices_ = np.array([x.item() if isinstance(x, np.ndarray) else x for x in Merged_DicesD])
    Ious_ = np.array([x.item() if isinstance(x, np.ndarray) else x for x in IousD])
    Merged_Ious_ = np.array([x.item() if isinstance(x, np.ndarray) else x for x in Merged_IousD])
    LDices_ = np.array([x.item() if isinstance(x, np.ndarray) else x for x in LDicesD])
    Merged_LDices_ = np.array([x.item() if isinstance(x, np.ndarray) else x for x in Merged_LDicesD])
    LIous_ = np.array([x.item() if isinstance(x, np.ndarray) else x for x in LIousD])
    Merged_LIous_ = np.array([x.item() if isinstance(x, np.ndarray) else x for x in Merged_LIousD])

    # Dices_, Merged_Dices_ = np.array(list(DicesD.values())),  np.array(list(Merged_DicesD.values()))  
    # Ious_, Merged_Ious_ = np.array(list(IousD.values())),  np.array(list(Merged_IousD.values()))
    # LDices_, Merged_LDices_ = np.array(list(LDicesD.values())), np.array(list(Merged_LDicesD.values()))
    # LIous_, Merged_LIous_ = np.array(list(LIousD.values())), np.array(list(Merged_LIousD.values()))

    mdice, merged_mdice = np.mean(Dices_), np.mean(Merged_Dices_)
    miou, merged_miou = np.mean(Ious_), np.mean(Merged_Ious_)
    mldice, merged_mldice = np.mean(LDices_), np.mean(Merged_LDices_)
    mliou, merged_mliou = np.mean(LIous_), np.mean(Merged_LIous_)
   
    return {"mdice":mdice, "loss": val_losses, "miou":miou, "mldice":mldice, "mliou": mliou,
            "merged_mdices": merged_mdice, "merged_miou": merged_miou, "merged_mldice": merged_mldice, "merged_liou":merged_mliou}


def eval_patient(valloader, model, opt):
    model.eval()
    val_losses = 0
    tps, fps, tns, fns = {}, {}, {}, {}

    for batch_idx, (datapack) in enumerate(valloader):

        with torch.no_grad():
            model_output = model(datapack)

        val_loss = model_output['loss']
        val_losses += val_loss.item()

        gt_mask = Variable(datapack['mask_merged'].to(device=opt.device))
        gt_mask = gt_mask.detach().cpu().numpy()

        if model_output['predict'].shape[1] == 2:
            predict = F.softmax(model_output['predict'], dim=1)
            predict = predict.detach().cpu().numpy()  # (b, c, h, w)
            seg = np.argmax(predict, axis=1)  # (b, h, w)
        else:
            predict = torch.sigmoid(model_output['predict'])
            predict = predict.detach().cpu().numpy() # (b, 1, h, w)
            seg = predict[:, 0, :, :] > 0.5  # (b, h, w)
        b, h, w = seg.shape

        for j in range(0, b):
            patient_key = datapack['patient_id'][j] + datapack['referring_txt'][j]
            pred_j = np.zeros((1, h, w))
            pred_j[seg[j:j+1, :, :] == 1] = 255
            gt_j = np.zeros((1, h, w))
            gt_j[gt_mask[j:j+1, :, :] == 1] = 255
            tp, fp, tn, fn = metrics.get_matrix(pred_j, gt_j)       
            tps[patient_key] = tps.get(patient_key, 0) + tp
            fps[patient_key] = fps.get(patient_key, 0) + fp
            tns[patient_key] = tns.get(patient_key, 0) + tn
            fns[patient_key] = fns.get(patient_key, 0) + fn
            if opt.visual:
                visual_segmentation(seg[j:j+1, :, :], datapack['image_path'][j], opt)
    
    val_losses = val_losses / (batch_idx + 1)
    tps = np.array(list(tps.values())) # In Python 3.7+, dict are ordered
    fps = np.array(list(fps.values()))
    tns = np.array(list(tns.values()))
    fns = np.array(list(fns.values()))

    smooth = 1e-5
    patient_dices = (2 * tps + smooth) / (2 * tps + fps + fns + smooth)  # [p]
    mdice = np.mean(patient_dices)  
    patient_iou = (tps + smooth) / (fps + tps + fns + smooth)
    miou = np.mean(patient_iou)
    patient_acc = (tps + tns + smooth)/(tps + fps + fns + tns + smooth)
    macc = np.mean(patient_acc, axis=0)
    patient_se = (tps + smooth) / (tps + fns + smooth)
    mse = np.mean(patient_se)
    patient_sp = (tns + smooth) / (fps + tns + smooth)
    msp = np.mean(patient_sp)
    return {"mdice":mdice, "loss": val_losses, "miou":miou, "macc":macc, "mse": mse, "sp": msp}


def eval_patient2(valloader, model, opt):
    model.eval()
    val_losses = 0
    tps, fps, tns, fns = {}, {}, {}, {}

    for batch_idx, (datapack) in enumerate(valloader):

        with torch.no_grad():
            model_output = model(datapack)

        val_loss = model_output['loss']
        val_losses += val_loss.item()

        gt_mask = Variable(datapack['mask_merged'].to(device=opt.device))
        gt_mask = gt_mask.detach().cpu().numpy()

        if model_output['predict'].shape[1] == 2:
            predict = F.softmax(model_output['predict'], dim=1)
            predict = predict.detach().cpu().numpy()  # (b, c, h, w)
            seg = np.argmax(predict, axis=1)  # (b, h, w)
        else:
            predict = torch.sigmoid(model_output['predict'])
            predict = predict.detach().cpu().numpy() # (b, 1, h, w)
            seg = predict[:, 0, :, :] > 0.5  # (b, h, w)
        b, h, w = seg.shape

        for j in range(0, b):
            patient_key = datapack['patient_id'][j] + "-"+ datapack['referring_txt'][j]
            pred_j = np.zeros((1, h, w))
            pred_j[seg[j:j+1, :, :] == 1] = 255
            gt_j = np.zeros((1, h, w))
            gt_j[gt_mask[j:j+1, :, :] == 1] = 255
            tp, fp, tn, fn = metrics.get_matrix(pred_j, gt_j)       
            tps[patient_key] = tps.get(patient_key, 0) + tp
            fps[patient_key] = fps.get(patient_key, 0) + fp
            tns[patient_key] = tns.get(patient_key, 0) + tn
            fns[patient_key] = fns.get(patient_key, 0) + fn
            if opt.visual:
                visual_segmentation(seg[j:j+1, :, :], datapack['image_path'][j], opt)
    
    val_losses = val_losses / (batch_idx + 1)
    tps = np.array(list(tps.values())) # In Python 3.7+, dict are ordered
    fps = np.array(list(fps.values()))
    tns = np.array(list(tns.values()))
    fns = np.array(list(fns.values()))

    smooth = 1e-5
    patient_dices = (2 * tps + smooth) / (2 * tps + fps + fns + smooth)  # [p]
    mdice = np.mean(patient_dices)  
    patient_iou = (tps + smooth) / (fps + tps + fns + smooth)
    miou = np.mean(patient_iou)
    patient_acc = (tps + tns + smooth)/(tps + fps + fns + tns + smooth)
    macc = np.mean(patient_acc, axis=0)
    patient_se = (tps + smooth) / (tps + fns + smooth)
    mse = np.mean(patient_se)
    patient_sp = (tns + smooth) / (fps + tns + smooth)
    msp = np.mean(patient_sp)
    return {"mdice":mdice, "loss": val_losses, "miou":miou, "macc":macc, "mse": mse, "sp": msp}



def eval_combined_slice2(val_loader, model, opt):
    model.eval()
    val_losses = 0
    merged_dices, merged_ious, merged_ldices, merged_lious = 0, 0, 0, 0
    combined_dices, combined_ious, combined_ldices, combined_lious = 0, 0, 0, 0
    eval_number = 0
    fpred_num = 0
    tpred_num = 0

    for batch_idx, (datapack) in enumerate(val_loader):

        with torch.no_grad():
            model_output = model(datapack)

        val_loss = model_output['loss']
        val_losses += val_loss.item()

        merged_gt = datapack['mask_merged'].to(dtype = torch.float32) # (b h w)
        merged_gt = merged_gt.detach().cpu().numpy() # (b h w)
        combined_gt = datapack['mask_combined'].to(dtype = torch.float32) # (b h w)
        combined_gt = combined_gt.detach().cpu().numpy() # (b h w)
        reference_existing = datapack['referring_existing'].to(dtype = torch.long).detach().cpu().numpy() #(b)
        referring_txt = datapack['referring_txt'] 
        obnum_gt = datapack['class_num'].to(dtype = torch.long).detach().cpu().numpy() #(b)

        pred_existing = model_output['pred_cquery'] # b q 2
        pred_existing = F.softmax(pred_existing, dim=-1) # b q 2
        pred_existing = pred_existing.detach().cpu().numpy() # (b, q, 2)
        pred_existing = np.argmax(pred_existing, axis=-1) # (b, q)

        pred_num = model_output['pred_num'].detach().cpu().numpy() # (b)

        pred_masks = model_output['pred_masks']  # (b q h w)
        pred_masks =  F.softmax(pred_masks, dim=1) # (b q h w)
        pred_masks = pred_masks.detach().cpu().numpy() # (b q h w)
        seg = np.argmax(pred_masks, axis=1)  # (b, h, w)

        b, h, w = seg.shape
        merged_pred = np.zeros((b, h, w))
        for i in range(0, b):
            pred_numi = round(pred_num[i])
            gt_numi = obnum_gt[i]

            if pred_numi == gt_numi:
                tpred_num = tpred_num + 1
            else:
                fpred_num = fpred_num + 1

            combined_dice, combined_iou, combined_ldice, combined_liou = 0, 0, 0, 0
            for c in range(1, gt_numi+1):
                pred_c = np.zeros((1, h, w))
                if pred_existing[i, c] > 0:
                    pred_c[seg[i:i+1, :, :] == c] = 1
                gt_c = np.zeros((1, h, w))
                gt_c[combined_gt[i:i+1, :, :] == c] = 1
                dice_c, iou_c, ldice_c, liou_c = metrics.combined_coefficient(pred_c, gt_c, reference_existing[i:i+1], pred_existing[i, c])
                combined_dice, combined_iou, combined_ldice, combined_liou =  combined_dice + dice_c, combined_iou + iou_c, combined_ldice + ldice_c, combined_liou +liou_c
            combined_dice, combined_iou, combined_ldice, combined_liou = combined_dice/gt_numi, combined_iou/gt_numi, combined_ldice/gt_numi, combined_liou/gt_numi
            combined_dices, combined_ious, combined_ldices, combined_lious =  combined_dices + combined_dice, combined_ious + combined_iou, combined_ldices + combined_ldice, combined_lious + combined_liou

            pred_existingi = pred_existing[i][:pred_numi+1] # (k)
            segi = seg[i, : , :] # (h w)
            segi[segi > pred_numi] = 0 
            for c in range(1, pred_numi+1):
                if pred_existingi[c] == 0:
                    segi[segi == c] = 0
            segi[segi > 0] = 1
            merged_pred[i, :, :] = segi

            # if opt.visual:
            #     text_refering = datapack['referring_txt'][i]
            #     class_index = datapack['class_index'][i]
            #     visual_segmentation(segi, datapack['image_path'][i], opt)
        
        predict = F.softmax(model_output['predict'], dim=1)
        predict = predict.detach().cpu().numpy() # (b, 2, h, w)
        merged_pred2 = np.argmax(predict, axis=1)  # (b, h, w)
        #print(np.sum(merged_pred), "  ", np.sum(merged_gt))

        merged_dice, merged_iou, merged_ldice, merged_liou = metrics.merged_coefficient(merged_pred, merged_gt, reference_existing, pred_num)
        
        merged_dices, merged_ious, merged_ldices, merged_lious = merged_dices + merged_dice, merged_ious + merged_iou, merged_ldices+merged_ldice, merged_lious+merged_liou

        eval_number = eval_number + b


    mean_merged_dice, mean_merged_iou, mean_merged_ldice, mean_merged_liou = merged_dices/eval_number, merged_ious/eval_number, merged_ldices/eval_number, merged_lious/eval_number
    
    mean_combined_dice, mean_combined_iou, mean_combined_ldice, mean_combined_liou = combined_dices/eval_number, combined_ious/eval_number, combined_ldices/eval_number, combined_lious/eval_number
    
    val_losses = val_losses / (batch_idx + 1)
    print("true pred", tpred_num, "false pred", fpred_num)
    
    return {"mdice":mean_merged_dice, "loss": val_losses, "miou":mean_merged_iou, "mldice":mean_merged_ldice, "mliou": mean_merged_liou,
            "merged_mdices":mean_combined_dice, "merged_miou":mean_combined_iou, "merged_mldice":mean_combined_ldice, "merged_liou":mean_combined_liou}


def eval_merged_slice(valloader, model, opt):
    model.eval()
    val_losses = 0
    merged_dices, merged_ious, merged_ldices, merged_lious = 0, 0, 0, 0
    eval_number = 0
    for batch_idx, (datapack) in enumerate(valloader):

        with torch.no_grad():
            model_output = model(datapack)

        val_loss = model_output['loss']
        val_losses += val_loss.item()

        merged_gt = datapack['mask_merged'].to(dtype = torch.float32, device=opt.device) # (b h w)
        merged_gt = merged_gt.detach().cpu().numpy() # (b h w)
        reference_existing = datapack['referring_existing'].to(dtype = torch.long, device=opt.device).detach().cpu().numpy() #(b)

        pred_existing = F.softmax(model_output['pred_cquery'], dim=-1) # b q 2
        pred_existing = pred_existing.detach().cpu().numpy() # (b, q, 2)
        pred_existing = np.argmax(pred_existing, axis=-1) # (b, q)

        pred_num = model_output['pred_num'].detach().cpu().numpy()

        pred_masks = torch.sigmoid(model_output['pred_masks']) # (b q h w)
        pred_masks = pred_masks.detach().cpu().numpy() # (b q h w)
        seg = pred_masks > 0.5  # (b q h w)

        b, q, h, w = seg.shape
        merged_pred = np.zeros((b, h, w))
        for i in range(0, b):
            pred_numi = round(pred_num[i])
            pred_existingi = pred_existing[i][:pred_numi] # (k)
            #print(pred_existingi)
            segi = seg[i, :pred_numi, :, :] # (k h w)
            #print("num", pred_numi)
            segi[pred_existingi==0, :, :] = 0
            segi = np.sum(segi, axis=0) #  (h w)
            segi[segi>=1] = 1
            merged_pred[i, :, :] = segi

            if opt.visual:
                text_refering = datapack['referring_txt'][i]
                class_index = datapack['class_index'][i]
                visual_segmentation(segi, datapack['image_path'][i], opt)
        
        predict = torch.sigmoid(model_output['predict']).squeeze(1)
        predict = predict.detach().cpu().numpy() # (b, h, w)
        merged_pred2 = predict[:, :, :] > 0.5  # (b, h, w)
        #print(np.sum(merged_pred), "  ", np.sum(merged_gt))
        merged_dice, merged_iou, merged_ldice, merged_liou = metrics.merged_coefficient(merged_pred, merged_gt, reference_existing, pred_num)
        
        merged_dices, merged_ious, merged_ldices, merged_lious = merged_dices + merged_dice, merged_ious + merged_iou, merged_ldices+merged_ldice, merged_lious+merged_liou

        eval_number = eval_number + b

    mean_merged_dice, mean_merged_iou, mean_merged_ldice, mean_merged_liou = merged_dices/eval_number, merged_ious/eval_number, merged_ldices/eval_number, merged_lious/eval_number
    val_losses = val_losses / (batch_idx + 1)
    
    return {"mdice":mean_merged_dice, "loss": val_losses, "miou":mean_merged_iou, "mldice":mean_merged_ldice, "mliou": mean_merged_liou,
            "merged_mdices":mean_merged_dice, "merged_miou":mean_merged_iou, "merged_mldice":mean_merged_ldice, "merged_liou":mean_merged_liou}


def eval_dam_slice(valloader, model, opt, visual_mode=0):
    model.eval()
    val_losses = 0
    merged_dices, merged_ious, merged_ldices, merged_lious = 0, 0, 0, 0
    combined_dices, combined_ious, combined_ldices, combined_lious = 0, 0, 0, 0
    mask_losses, cls_losses, num_losses = 0, 0, 0
    matching_losses, merged_losses = 0, 0
    eval_number = 0
    fpred_num = 0
    tpred_num = 0
    GRecall, GPrecision = 0, 0
    thresh = 0.5
    dices_no, dices_multi, iou_no, iou_multi = 0, 0, 0, 0
    num_no, num_multi = 0, 0

    keep_excel = []

    for batch_idx, (datapack) in enumerate(valloader):

        with torch.no_grad():
            model_output = model(datapack)

        val_loss = model_output['loss']
        val_losses += val_loss.item()

        loss_dict = model_output["loss_dict"]
        loss_mask, loss_cls, loss_num = loss_dict["loss_masks"], loss_dict["loss_class"], loss_dict["loss_num"]
        loss_matching, loss_merged = loss_dict["loss_matching"], loss_dict["loss_merged"]
        mask_losses, cls_losses, num_losses = mask_losses+loss_mask.item(), cls_losses+loss_cls.item(), num_losses+loss_num.item()
        matching_losses, merged_losses = matching_losses + loss_matching.item(), merged_losses + loss_merged.item()

        merged_gt = datapack['mask_merged'].to(dtype = torch.float32) # (b h w)
        merged_gt = merged_gt.detach().cpu().numpy() 
        masks_gt = datapack['mask_instances'].to(dtype = torch.float32) # (b q h w)
        masks_gt = masks_gt.detach().cpu().numpy() 
        reference_existing = datapack['referring_existing'].to(dtype = torch.long).detach().cpu().numpy() #(b)
        gt_existing = datapack['mask_existing'].to(dtype = torch.long).detach().cpu().numpy() #(b q)
        referring_txt = datapack['referring_txt'] 
        obnum_gt = datapack['class_num'].to(dtype = torch.long).detach().cpu().numpy() #(b)

        pred_existing = model_output['pred_cquery'] # b q 2
        pred_existing = F.softmax(pred_existing, dim=-1) # b q 2
        pred_existing = pred_existing.detach().cpu().numpy() # (b, q, 2)
        pred_existing = np.argmax(pred_existing, axis=-1) # (b, q)

        pred_num = model_output['pred_num'].detach().cpu().numpy() # (b)

        # --------------------------------------------------------------------
        pred_masks = torch.sigmoid(model_output['pred_masks']) # (b q h w)
        pred_masks = pred_masks.detach().cpu().numpy() # (b q h w)
        seg = pred_masks > 0.5  # (b q h w)
        # ----------------------------------------------------------------

        b, q, h, w = seg.shape
        merged_pred = np.zeros((b, h, w))
        for i in range(0, b):
            pred_numi = round(pred_num[i])
            gt_numi = obnum_gt[i]

            N_den = max(pred_numi, gt_numi)
            N_num = min(pred_numi, gt_numi)
            alpha = (N_num + 1e-4)/(N_den + 1e-4)

            if pred_numi == gt_numi:
                tpred_num = tpred_num + 1
            else:
                fpred_num = fpred_num + 1

            combined_dice, combined_iou, combined_ldice, combined_liou = 0, 0, 0, 0
            Otp, Op, Og = 0, 0, 0
            pred_any_masks = []
            for c in range(gt_numi):
                pred_c = np.zeros((1, h, w))
                if pred_existing[i, c] > 0:
                    pred_c[seg[i:i+1, c, :, :] == 1] = 1
                gt_c = np.zeros((1, h, w))
                gt_c[masks_gt[i:i+1, c, :, :] == 1] = 1
                dice_c, iou_c, ldice_c, liou_c = metrics.combined_coefficient(pred_c, gt_c, reference_existing[i:i+1], pred_existing[i, c])
                combined_dice, combined_iou, combined_ldice, combined_liou =  combined_dice + dice_c, combined_iou + iou_c, combined_ldice + ldice_c, combined_liou +liou_c

                if gt_existing[i, c]:
                    Og = Og + 1
                    if dice_c > thresh:
                        Otp = Otp + 1
                if np.any(pred_c):
                        Op = Op + 1

                pred_any_masks.append(pred_c)
                # if visual_mode == 1:
                #     visual_segmentation_single(pred_c, datapack['image_path'][i], datapack['image_name'][i], datapack['referring_txt'][i], str(c))

            recall = alpha * ((Otp + 1e-4) / (Og + 1e-4))
            precision = alpha * ((Otp + 1e-4) / (Op + 1e-4))
            GRecall, GPrecision = GRecall + recall, GPrecision + precision

            if visual_mode == 2:
                #visual_segmentation_any(pred_any_masks, datapack['image_path'][i], datapack['image_name'][i], datapack['class_index'][i])
                visual_segmentation_order(pred_any_masks, datapack['image_path'][i], datapack['image_name'][i], datapack['referring_txt'][i], datapack['class_index'][i], combined_dice/gt_numi)

            #visual_image_heatmap(model_output['pred_masks'][i, :, :, :], datapack['image_path'][i], datapack['image_name'][i], datapack['class_index'][i])

            combined_dice, combined_iou, combined_ldice, combined_liou = combined_dice/gt_numi, combined_iou/gt_numi, combined_ldice/gt_numi, combined_liou/gt_numi
            combined_dices, combined_ious, combined_ldices, combined_lious =  combined_dices + combined_dice, combined_ious + combined_iou, combined_ldices + combined_ldice, combined_lious + combined_liou

            pred_existingi = pred_existing[i][:pred_numi] # (k)
            segi = seg[i, :pred_numi, :, :] # (k h w)
            segi[pred_existingi==0, :, :] = 0
            segi = np.sum(segi, axis=0) #  (h w)
            segi[segi>=1] = 1
            merged_pred[i, :, :] = segi

            mdicei, mioui, _, _ = metrics.merged_coefficient(merged_pred[i:i+1, :, :], merged_gt[i:i+1, :, :], reference_existing, pred_num)

            if not datapack['referring_existing'][i]:
                num_no = num_no + 1
                dices_no = dices_no + mdicei
                iou_no = iou_no + mioui
            
            if datapack['class_num'][i] > 1:
                num_multi = num_multi+1
                dices_multi = dices_multi + mdicei
                iou_multi = iou_multi + mioui
            
        merged_dice, merged_iou, merged_ldice, merged_liou = metrics.merged_coefficient(merged_pred, merged_gt, reference_existing, pred_num)
        
        merged_dices, merged_ious, merged_ldices, merged_lious = merged_dices + merged_dice, merged_ious + merged_iou, merged_ldices+merged_ldice, merged_lious+merged_liou

        eval_number = eval_number + b

    mean_merged_dice, mean_merged_iou, mean_merged_ldice, mean_merged_liou = merged_dices/eval_number, merged_ious/eval_number, merged_ldices/eval_number, merged_lious/eval_number
    mean_combined_dice, mean_combined_iou, mean_combined_ldice, mean_combined_liou = combined_dices/eval_number, combined_ious/eval_number, combined_ldices/eval_number, combined_lious/eval_number
    mean_grecall, mean_gprecision = GRecall/eval_number, GPrecision/eval_number

    val_losses = val_losses / (batch_idx + 1)

    # mean_dice_no = dices_no/num_no
    # mean_iou_no = iou_no/ num_no
    # mean_dice_multi = dices_multi/num_multi
    # mean_iou_multi = iou_multi/num_multi
    # print("num no:", num_no, "  num multi:", num_multi)
    # print("dice_no:", mean_dice_no, "iou_no;", mean_iou_no, "dice_multi:", mean_dice_multi, "iou_multi:", mean_iou_multi)

    # keep_excel = pd.DataFrame(keep_excel)
    # keep_excel.to_excel('dam2.xlsx', index=False)
    
    return {"mdice":mean_merged_dice, "loss": val_losses, "miou":mean_merged_iou, "mldice":mean_merged_ldice, "mliou": mean_merged_liou,
            "combined_mdices":mean_combined_dice, "combined_miou":mean_combined_iou, "Grecall": mean_grecall, "Gprecision":mean_gprecision}


def get_eval(valloader, model, opt, args):
    if opt.eval_mode == "slice":
        return eval_slice(valloader, model, opt)
    elif opt.eval_mode == "dynamic_masks_slice":
        return eval_dm_slice(valloader, model, opt)
    elif opt.eval_mode == "merged_slice":
        return eval_merged_slice(valloader, model, opt)
    elif opt.eval_mode == "dam_slice":
        return eval_dam_slice(valloader, model, opt)
    elif opt.eval_mode == "combined_slice":
        return eval_combined_slice2(valloader, model, opt)
    elif opt.eval_mode == "patient":
        return eval_patient(valloader, model, opt)
    elif opt.eval_mode == "dynamic_masks_patient":
        return eval_dm_patient(valloader, model, opt)
    else:
        raise RuntimeError("Could not find the eval mode:", opt.eval_mode)