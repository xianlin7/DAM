# -*- coding: utf-8 -*-
import os
import torch
import time
import ml_collections

n_channels = 1
img_size = 256 
task_name = 'GMIRS' 
model_name = 'DAM'
max_tokens = 25
num_query = 9

##########################################################################
# Model configs
##########################################################################

def get_model_config():
    config = ml_collections.ConfigDict()
    config.transformer = ml_collections.ConfigDict()
    config.n_channels = n_channels
    config.token_length = max_tokens
    config.num_query = num_query
    config.base_channel = 64  # base channel of U-Net
    config.img_size = img_size
    config.align_window_size = 8
    return config


# used in testing phase, copy the session name in training phase

