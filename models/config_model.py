
# This file is used to obtain the training hyperparameters of different models

#from transformers.models.bert.modeling_bert import BertModel
import torch
from functools import reduce
import operator

class Config_DAM_MosMed:
    optimizer = "adam" #"adamw" # "adam"
    weight_decay =  1e-5  # 0.01 # 1e-5
    lr = 1e-4 # 3e-4
    epochs = 300   
    batch_size = 4    
    num_workers = 8
    image_size = 224
    lr_scheduler = "cosineLR" # "polynomial" # "constant"
    early_stopping_patience = 50
    # periodic = [100, 200]
    # periodic_decay = 0.1

    def get_optimizer(self, model, lr=5e-5, weight_decay=0.01):
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
        #optimizer = torch.optim.AdamW(list(model.parameters()), lr=lr, weight_decay=weight_decay, amsgrad=False)
        return optimizer

class Config_DAM:
    optimizer = "adam" #"adamw" # "adam"
    weight_decay =  1e-5  # 0.01 # 1e-5
    lr = 1e-4 # 3e-4
    epochs = 200   
    batch_size = 16    
    num_workers = 8
    image_size = 256
    lr_scheduler = "cosineLR" # "polynomial" # "constant"
    early_stopping_patience = 50
    # periodic = [100, 200]
    # periodic_decay = 0.1

    def get_optimizer(self, model, lr=5e-5, weight_decay=0.01):
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
        #optimizer = torch.optim.AdamW(list(model.parameters()), lr=lr, weight_decay=weight_decay, amsgrad=False)
        return optimizer
    

def get_model_config(modelname="DAM_GMIRS"):
    if modelname == "DAM_MosMed":
        return Config_DAM_MosMed()
    elif modelname == "DAM_GMIRS":
        return Config_DAM()
    else:
        assert("No corresponding training hyperparameters were found for this model.")

