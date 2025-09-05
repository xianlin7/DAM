# -*- coding: utf-8 -*-
import os
import torch
import time
import ml_collections

## PARAMETERS OF THE MODEL
save_model = True
tensorboard = True
use_cuda = torch.cuda.is_available()
seed = 10001 #666
os.environ['PYTHONHASHSEED'] = str(seed)

cosineLR = True  # Use cosineLR or not
#lr_scheduler = "exp"

n_channels = 1
n_labels = 1  # MoNuSeg & Covid19
epochs = 2000
img_size = 224
print_frequency = 1
save_frequency = 5000
vis_frequency = 10
early_stopping_patience = 50

pretrain = False
task_name = 'MosMedDataPlus' 
learning_rate = 1e-4
batch_size = 8

model_name = 'DAM4AlignUp'

max_tokens = 18
num_query = 2


train_dataset = './datasets/' + task_name + '/Train_Folder/'
val_dataset = './datasets/' + task_name + '/Val_Folder/'
test_dataset = './datasets/' + task_name + '/Test_Folder/'
task_dataset = './datasets/' + task_name + '/Train_Folder/'
session_name = 'Test_session' + '_dim192token18vit2b85e-4sd10001_' + time.strftime('%m.%d_%Hh%M')
save_path = './checkpoints/' + task_name + '/' + model_name + '/' + session_name + '/'
model_path = save_path + 'models/'
tensorboard_folder = save_path + 'tensorboard_logs/'
logger_path = save_path + session_name + ".log"
visualize_path = save_path + 'visualize_val/'


##########################################################################
# Model configs
##########################################################################

def get_model_config():
    config = ml_collections.ConfigDict()
    config.transformer = ml_collections.ConfigDict()
    config.n_channels = n_channels
    config.token_length = max_tokens
    config.num_query = num_query
    config.img_size = img_size
    config.align_window_size = 7
    config.base_channel = 64  # base channel of U-Net
    config.n_classes = 1
    return config


# used in testing phase, copy the session name in training phase

#test_session = "Test_session_adapter2_02.21_11h14" # dice=73.75, IoU=60.46
test_session = "Test_session_dim192_02.27_10h28" # Bdice=74.42, IoU=61.53

test_session = "Test_session_dim192token18_02.28_15h41" # dice=74.96, IoU=62.02
