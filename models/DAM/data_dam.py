import os
from random import randint
import numpy as np
import torch
from skimage import io, color
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F
from typing import Callable
import os
import cv2
import pandas as pd
from numbers import Number
from typing import Container
from collections import defaultdict
from collections import OrderedDict
from torchvision.transforms import InterpolationMode
from transformers import BertTokenizer
from bert_embedding import BertEmbedding


def to_long_tensor(pic):
    # handle numpy array
    img = torch.from_numpy(np.array(pic, np.uint8))
    # backward compatibility
    return img.long()


def correct_dims(*images):
    corr_images = []
    for img in images:
        if len(img.shape) == 2:
            corr_images.append(np.expand_dims(img, axis=2))
        else:
            corr_images.append(img)
    if len(corr_images) == 1:
        return corr_images[0]
    else:
        return corr_images

class JointTransform2D:
    """
    Performs augmentation on image and mask when called. Due to the randomness of augmentation transforms,
    it is not enough to simply apply the same Transform from torchvision on the image and mask separetely.
    Doing this will result in messing up the ground truth mask. To circumvent this problem, this class can
    be used, which will take care of the problems above.
    """

    def __init__(self, img_size=256, crop=(32, 32), p_rota=0.0, p_scale=0.0, p_gaussn=0.0, p_contr=0.0,
                 p_gama=0.0, p_distor=0.0, color_jitter_params=(0.1, 0.1, 0.1, 0.1), p_random_affine=0,
                 long_mask=False):
        self.crop = crop
        self.p_rota = p_rota
        self.p_scale = p_scale
        self.p_gaussn = p_gaussn
        self.p_gama = p_gama
        self.p_contr = p_contr
        self.p_distortion = p_distor
        self.img_size = img_size
        self.color_jitter_params = color_jitter_params
        if color_jitter_params:
            self.color_tf = T.ColorJitter(*color_jitter_params)
        self.p_random_affine = p_random_affine
        self.long_mask = long_mask

    def __call__(self, image, mask):
        #  gamma enhancement
        if np.random.rand() < self.p_gama:
            c = 1
            g = np.random.randint(10, 25) / 10.0
            # g = 2
            image = (np.power(image / 255, 1.0 / g) / c) * 255
            image = image.astype(np.uint8)
        # transforming to PIL image
        image, mask = F.to_pil_image(image), F.to_pil_image(mask)
        # random crop
        if self.crop:
            i, j, h, w = T.RandomCrop.get_params(image, self.crop)
            image, mask = F.crop(image, i, j, h, w), F.crop(mask, i, j, h, w)
        # random rotation
        if np.random.rand() < self.p_rota:
            angle = T.RandomRotation.get_params((-15, 15))
            image, mask = F.rotate(image, angle), F.rotate(mask, angle)
        # random scale and center resize to the original size
        if np.random.rand() < self.p_scale:
            scale = np.random.uniform(1, 1.3)
            new_h, new_w = int(self.img_size * scale), int(self.img_size * scale)
            image, mask = F.resize(image, (new_h, new_w), InterpolationMode.BILINEAR), F.resize(mask, (new_h, new_w), InterpolationMode.NEAREST)
            # image = F.center_crop(image, (self.img_size, self.img_size))
            # mask = F.center_crop(mask, (self.img_size, self.img_size))
            i, j, h, w = T.RandomCrop.get_params(image, (self.img_size, self.img_size))
            image, mask = F.crop(image, i, j, h, w), F.crop(mask, i, j, h, w)
        # random add gaussian noise
        if np.random.rand() < self.p_gaussn:
            ns = np.random.randint(3, 15)
            noise = np.random.normal(loc=0, scale=1, size=(self.img_size, self.img_size)) * ns
            noise = noise.astype(int)
            image = np.array(image) + noise
            image[image > 255] = 255
            image[image < 0] = 0
            image = F.to_pil_image(image.astype('uint8'))
        # random change the contrast
        if np.random.rand() < self.p_contr:
            contr_tf = T.ColorJitter(contrast=(0.8, 2.0))
            image = contr_tf(image)
        # random distortion
        if np.random.rand() < self.p_distortion:
            distortion = T.RandomAffine(0, None, None, (5, 30))
            image = distortion(image)
        # color transforms || ONLY ON IMAGE
        if self.color_jitter_params:
            image = self.color_tf(image)
        # random affine transform
        if np.random.rand() < self.p_random_affine:
            affine_params = T.RandomAffine(180).get_params((-90, 90), (1, 1), (2, 2), (-45, 45), self.crop)
            image, mask = F.affine(image, *affine_params), F.affine(mask, *affine_params)
        # transforming to tensor
        image, mask = F.resize(image, (self.img_size, self.img_size), InterpolationMode.BILINEAR), F.resize(mask, (self.img_size, self.img_size), InterpolationMode.NEAREST)
        image = F.to_tensor(image)

        if not self.long_mask:
            mask = F.to_tensor(mask)
        else:
            mask = to_long_tensor(mask)
        return image, mask


class ImageToImage2D(Dataset):
    """
    Reads the images and applies the augmentation transform on them.

    Args:
        dataset_path: path to the dataset. Structure of the dataset should be:
            dataset_path
                |-- MainPatient
                    |-- train.xlsx
                    |-- val.xlsx
                    |-- text.xlsx
                |-- subtask1
                    |-- img
                        |-- patientid_sliceid.png
                        |-- 001_002.png
                        |-- ...
                    |-- label
                        |-- patientid_sliceid.png
                        |-- 001_002.png
                        |-- ...
                |-- subtask2
                    |-- img
                        |-- patientid_sliceid.png
                        |-- 001_002.png
                        |-- ...
                    |-- label
                        |-- patientid_sliceid.png
                        |-- 001_002.png
                        |-- ...
                |-- subtask...   

        joint_transform: augmentation transform, an instance of JointTransform2D. If bool(joint_transform)
            evaluates to False, torchvision.transforms.ToTensor will be used on both image and mask.
        one_hot_mask: bool, if True, returns the mask in one-hot encoded form.
    """

    def __init__(self, dataset_path: str, split='train', joint_transform: Callable = None, img_size=256, max_tokens=25, max_query=9, one_hot_mask: int = False) -> None:

        self.dataset_path = dataset_path
        self.one_hot_mask = one_hot_mask
        self.split = split
        information_file = os.path.join(dataset_path, 'MainPatient/{0}.xlsx'.format(split))
        information = pd.read_excel(information_file)
        # note: the column headings of the Excel file are ["Dataset Name", "Class Index", "Image Name", "Description"]
        self.ids = information.to_dict('records')
        self.img_size = img_size
        self.max_tokens = max_tokens
        self.max_query = max_query

        self.bert_embedding = BertEmbedding()

        if joint_transform:
            self.joint_transform = joint_transform
        else:
            to_tensor = T.ToTensor()
            self.joint_transform = lambda x, y: (to_tensor(x), to_tensor(y))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        id_ = self.ids[i]
        sub_dataset, class_index, filename, referring_txt = id_["Dataset Name"], id_["Class Index"], id_["Image Name"], id_["Description"]

        # referring_txt = "Liver, spleen, left kiney, right kidney, stomach, gallbladder, and pancreas."
        #class_index = "7-9-10-11-12-13-14-15"
        

        #referring_txt = "COVID-19 contains bilateral pulmonary infection, two infected areas, left lung and right lung."

        img_path = os.path.join(os.path.join(self.dataset_path, sub_dataset), 'img')
        label_path = os.path.join(os.path.join(self.dataset_path, sub_dataset), 'label')

        image = cv2.imread(os.path.join(img_path, filename), 0)
        mask = cv2.imread(os.path.join(label_path, filename), 0)

        if sub_dataset == "INSTANCE" or sub_dataset == "Adrenal-ACC-Ki67-53" or sub_dataset == "Covid-19-20" or sub_dataset == "MosMedDataPlus" or sub_dataset == "QaTa-Covid19":
            mask[mask > 1] = 1

        # correct dimensions if needed
        image, mask = correct_dims(image, mask)  
        if self.joint_transform:
            image, mask = self.joint_transform(image, mask)
        
        if self.one_hot_mask:
            assert self.one_hot_mask > 0, 'one_hot_mask must be nonnegative'
            mask = torch.zeros((self.one_hot_mask, mask.shape[1], mask.shape[2])).scatter_(0, mask.long(), 1)

        mask_shape = mask.shape
        mask_instances = torch.zeros((self.max_query, *mask_shape))
        mask_existing =  torch.zeros((self.max_query))
        mask_merged = torch.zeros_like(mask)
        mask_combined = torch.zeros_like(mask)
    
        classes = str(class_index).split("-")
        class_num = len(classes)
        referring_existing = 0
        for i in range(class_num):
            mask_i = mask.clone()
            mask_i[mask_i != int(classes[i])] = 0
            mask_i[mask_i == int(classes[i])] = 1
            mask_instances[i, :, :] =  mask_i
            mask_merged[mask_i == 1] = 1
            mask_combined[mask_i == 1] = i+1
            if torch.any(mask_i != 0):
                referring_existing = 1
                mask_existing[i] = 1
        
        # --------------------------------- Language data ---------------------------------

        text = referring_txt.split('\n')
        text_token = self.bert_embedding(text)
        text_token = np.array(text_token[0][1])
        text_token = text_token[:self.max_tokens, :]

        text_mask = [0] * self.max_tokens
        padded_text_tokens = np.zeros((self.max_tokens, text_token.shape[1]), dtype=text_token.dtype)
        text_mask[:text_token.shape[0]] = [1] * len(text_token)
        padded_text_tokens[:text_token.shape[0], :] = text_token
        text_mask = torch.tensor(text_mask)
        padded_text_tokens = torch.Tensor(padded_text_tokens)

        padded_text_label = [-1] * self.max_tokens
        padded_text_label =  torch.tensor(padded_text_label)
        text_label = id_["Text Label"]
        text_label = text_label.split('-')
        text_label = [int(x) for x in text_label]
        text_label = torch.tensor(text_label)
        text_label = text_label[:self.max_tokens]
        padded_text_label[:text_label.shape[0]] = text_label

        return {
            'image': image,
            'referring_txt': referring_txt,
            'mask_instances': mask_instances,
            'mask_merged': mask_merged,
            'mask_combined': mask_combined,
            'mask_existing': mask_existing,
            'lang_token': padded_text_tokens,
            'lang_mask': text_mask,
            'text_label': padded_text_label,
            'class_index': str(class_index),
            'class_num': class_num,
            'referring_existing': referring_existing,
            'image_name': filename,
            'image_path': os.path.join(img_path, filename),
            'data_name': sub_dataset,
            'patient_id': sub_dataset + "-" + filename.split('_')[0],
            'label_path': label_path,
            }

