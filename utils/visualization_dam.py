import torchvision
import os
import torch
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

def visual_segmentation_single(seg, image_path, filename, referring, object_id):
    img_ori0 = cv2.imread(os.path.join(image_path))
    img_ori = img_ori0.copy()
    overlay = img_ori * 0
    img_r = img_ori[:, :, 0]
    img_g = img_ori[:, :, 1]
    img_b = img_ori[:, :, 2]
    table = np.array([[193, 182, 255], [219, 112, 147], [237, 149, 100], [211, 85, 186], [144, 255, 144], [204, 209, 72],
                             [0, 215, 255], [96, 164, 244], [128, 128, 240], [250, 206, 135]])
    seg = seg[0, :, :]
    
    i = int(object_id)

    img_r[seg == 1] = table[i, 0]
    img_g[seg == 1] = table[i, 1]
    img_b[seg == 1] = table[i, 2]
            
    overlay[:, :, 0] = img_r
    overlay[:, :, 1] = img_g
    overlay[:, :, 2] = img_b
    overlay = np.uint8(overlay)
   
    img = cv2.addWeighted(img_ori0, 0.5, overlay, 0.5, 0) 
          
    fulldir = "visualization" + "/" + "compare_multi" + "/" + "RecLMSI" + "/"
    if not os.path.isdir(fulldir):
        os.makedirs(fulldir)
    cv2.imwrite(fulldir + filename.split('.')[0]  + "_" + referring + "_" + object_id + ".png", img)


def visual_segmentation_any(segs, image_path, filename, referring):
    img_ori0 = cv2.imread(os.path.join(image_path))
    img_ori = img_ori0.copy()
    overlay = img_ori * 0
    img_r = img_ori[:, :, 0]
    img_g = img_ori[:, :, 1]
    img_b = img_ori[:, :, 2]
    table = np.array([[193, 182, 255], [219, 112, 147], [237, 149, 100], [204, 209, 72], [144, 255, 144], [211, 85, 186],
                              [0, 215, 255], [96, 164, 244], [128, 128, 240], [250, 206, 135]])
    
    for i in range(len(segs)):
        segi = segs[i][0, :, :]
        img_r[segi == 1] = table[i, 0]
        img_g[segi == 1] = table[i, 1]
        img_b[segi == 1] = table[i, 2]
            
    overlay[:, :, 0] = img_r
    overlay[:, :, 1] = img_g
    overlay[:, :, 2] = img_b
    overlay = np.uint8(overlay)
   
    img = cv2.addWeighted(img_ori0, 0.5, overlay, 0.5, 0) 
          
    #fulldir = "visualization" + "/" + "compare_multi" + "/" + "DAM" + "/"
    fulldir = "visualization" + "/" + "word_gt" + "/"
    if not os.path.isdir(fulldir):
        os.makedirs(fulldir)
    cv2.imwrite(fulldir + filename.split('.')[0]  + "_" + referring + ".png", img)


def visual_segmentation_order(segs, image_path, filename, referring, class_index, dice=0):
    img_ori0 = cv2.imread(os.path.join(image_path))
    img_ori = img_ori0.copy()
    overlay = img_ori * 0
    img_r = img_ori[:, :, 0]
    img_g = img_ori[:, :, 1]
    img_b = img_ori[:, :, 2]
    table = np.array([[193, 182, 255], [219, 112, 147], [128, 128, 240], [204, 209, 72], [144, 255, 144], [211, 85, 186],
                              [0, 215, 255], [96, 164, 244], [237, 149, 100], [250, 206, 135]])
    class_id = class_index.split('-')
    for i in range(len(segs)):
        segi = segs[i][0, :, :]
        color_id = int(class_id[i])-1
        if color_id == (10-1):
            color_id = 6-1
        if color_id == (12-1):
            color_id = 9-1
        if color_id == (14-1):
            color_id = 9-1
        img_r[segi == 1] = table[color_id, 0]
        img_g[segi == 1] = table[color_id, 1]
        img_b[segi == 1] = table[color_id, 2]
            
    overlay[:, :, 0] = img_r
    overlay[:, :, 1] = img_g
    overlay[:, :, 2] = img_b
    overlay = np.uint8(overlay)
   
    img = cv2.addWeighted(img_ori0, 0.5, overlay, 0.5, 0) 
          
    #fulldir = "visualization" + "/" + "compare_multi" + "/" + "DAM" + "/"
    fulldir = "visualization" + "/" + "word_pred" + "/"
    if not os.path.isdir(fulldir):
        os.makedirs(fulldir)
    cv2.imwrite(fulldir + filename.split('.')[0]  + "_" + str(dice) + "_" + referring + ".png", img)


def visual_image_heatmap(heatmap, image_path, filename, referring):
    "heatmap:(q h w)"
    heatmap = torch.sigmoid(heatmap)
    image = cv2.imread(image_path)
    heatmap = heatmap.cpu().numpy()
    q =  heatmap.shape[0]
    for i in range(q):
        heatmapi = heatmap[i, :, :]
        #heatmapi = (heatmapi - heatmapi.min() + 1e-4) / (heatmapi.max() - heatmapi.min() + 1e-4)
        heatmapi = np.uint8(255 * heatmapi)
        heatmapi = cv2.applyColorMap(heatmapi, cv2.COLORMAP_JET)

        image_hmi = cv2.addWeighted(image, 0.6, heatmapi, 0.4, 0)
                
        fulldir = "visualization" + "/" + "heatmap_image_word" + "/"
        if not os.path.isdir(fulldir):
            os.makedirs(fulldir)
        cv2.imwrite(fulldir + filename.split('.')[0]  + "_" + referring + "_" + str(i) + ".png", image_hmi)


def visual_text_heatmap(heatmap, filename, referring):
    "heatmap:(q t)"

    heatmap = heatmap.cpu().numpy()
    t, q =  heatmap.shape[0], heatmap.shape[1]
    
    heatmapi = (heatmap - heatmap.min() + 1e-4) / (heatmap.max() - heatmap.min() + 1e-4)

    #heatmapi = np.clip(heatmapi, 0.3, 0.4)
    #heatmapi = np.uint8(255 * heatmap)

    heatmapi = cv2.normalize(heatmapi, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    heatmapi = cv2.applyColorMap(heatmapi, cv2.COLORMAP_JET) 

    x = np.zeros((t*10, q*30, 3))+255
    for i in range(t):
        for j in range(q):
            x[i*10:(i*10+7), j*30:(j*30+25), :] = heatmapi[i, j, :]
    fheatmap = np.uint8(x)

    
                
    fulldir = "visualization" + "/" + "heatmap_text_word" + "/"
    if not os.path.isdir(fulldir):
        os.makedirs(fulldir)
    cv2.imwrite(fulldir + filename.split('.')[0]  + "_" + referring + ".png", fheatmap)