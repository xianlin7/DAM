import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import time
import random
from utils.config import get_task_config
from models.model_dict import get_model
import torch.optim as optim
from utils.loss_functions.dice_loss import DC_and_BCE_loss, DC_and_CE_loss
from utils.evaluation import get_eval
from thop import profile


def main():

    #  =========================================== parameters setting ==================================================

    parser = argparse.ArgumentParser(description='Networks')
    parser.add_argument('--modelname', default='DAM_GMIRS', type=str, help='type of model, e.g., LAVT, VLT...')
    parser.add_argument('-image_size', type=int, default=256, help='the input image size') 
    parser.add_argument('--task', default='GMIRS', help='task or dataset name')
    parser.add_argument('--max_tokens', type=int, default=25, help='batch_size per gpu') 
    parser.add_argument('--batch_size', type=int, default=8, help='batch_size per gpu') 
    parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
    parser.add_argument('--base_lr', type=float, default=0.0001, help='segmentation network learning rate') 
    parser.add_argument('-keep_log', type=bool, default=True, help='keep the loss&lr&dice during training or not')
    parser.add_argument('-use_custom_hyper', type=bool, default=True, help='Use user specified training hyperparameters')
    parser.add_argument('-lr_scheduler', default='constant', help='Learning rate scheduler type')
    parser.add_argument('-load_path', default='./checkpoints/DAM_GMIRS.pth', help='checkpoint path of the trained model') 
    args = parser.parse_args()
    
    opt = get_task_config(args.task)  # please configure your hyper-parameter
    opt.eval_mode = "dam_slice"
    opt.batch_size = 8
   
    #  ============================= add the seed to make sure the results are reproducible ============================

    seed_value = 1234  # the number of seed
    np.random.seed(seed_value)  # set random seed for numpy
    random.seed(seed_value)  # set random seed for python
    os.environ['PYTHONHASHSEED'] = str(seed_value)  # avoid hash random
    torch.manual_seed(seed_value)  # set random seed for CPU
    torch.cuda.manual_seed(seed_value)  # set random seed for one GPU
    torch.cuda.manual_seed_all(seed_value)  # set random seed for all GPU
    torch.backends.cudnn.deterministic = True  # set random seed for convolution

    #  ========================================= model and data initialization ==========================================
    
    # register the sam model
    model, model_optimizer, mconfig = get_model(args.modelname, args=args, opt=opt)

    if args.modelname == "DAM_GMIRS" or args.modelname == "DAM_MosMed" or args.modelname == "DAM":
        from models.DAM.data_dam import JointTransform2D, ImageToImage2D
    else:
        from utils.data_gmirs import JointTransform2D, ImageToImage2D
    
    if args.task == "MosMed" or args.task == "QaTaCovid19":
        from utils.data_SimpleTokenize import LViTGenerator as JointTransform2D

    tf_test = JointTransform2D(img_size=opt.img_size, crop=opt.crop, color_jitter_params=None, long_mask=True)
    test_dataset = ImageToImage2D(opt.data_path, opt.test_split, tf_test, img_size=opt.img_size, max_tokens=args.max_tokens)  # return image, mask, and filename
    testloader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers, pin_memory=True)

    device = torch.device(opt.device)
    model.to(device)
    checkpoint = torch.load(args.load_path)
    model.load_state_dict(checkpoint)
    print("load the model:", args.load_path)
   
   # -------------------------------------------------------------------------------
    # pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print("Total_params: {}".format(pytorch_total_params))

    input = torch.randn(1, 3, 224, 224)
    label =  torch.randint(low=0, high=1, size=(1, 224, 224)).long()
    #text = torch.randn(1, 10, 768)
    text = torch.tensor([0]*10)
    text_mask =  torch.zeros(1, 10)
    text_mask[:, :5] = 1
    
    # data = {'image': input, 'label': label, 'lang_token': text, 'lang_mask': text_mask}
    # flops, params = profile(model, inputs=(data, "cuda", ))
    # print('flops:{}'.format(flops/1000000000))
    # print('params:{}'.format(params/1000000))
    # -----------------------------------------------------------------------------

    model.eval()
    val_output = get_eval(testloader, model, opt=opt, args=args)
    print("------ testing result of dataset:" + args.task + " ------ model name: "+ args.modelname + "------")
    print("mdice:", val_output['mdice'], "miou:", val_output['miou'])
    print(val_output)

if __name__ == '__main__':
    main()