import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
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
from utils.lr_scheduler import CosineAnnealingWarmRestarts
from utils.evaluation import get_eval


def main():

    #  =========================================== parameters setting ==================================================

    parser = argparse.ArgumentParser(description='Networks')
    parser.add_argument('--modelname', default="DAM_GMIRS", type=str, help='type of model, e.g., LAVT, VLT, LViT, SLViT...')
    parser.add_argument('-image_size', type=int, default=256, help='the input image size') 
    parser.add_argument('--task', default='GMIRS', help='task or dataset name')
    parser.add_argument('--max_tokens', type=int, default=25, help='batch_size per gpu') 
    parser.add_argument('--batch_size', type=int, default=8, help='batch_size per gpu') 
    parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
    parser.add_argument('--base_lr', type=float, default=0.0001, help='segmentation network learning rate') 
    parser.add_argument('-keep_log', type=bool, default=True, help='keep the loss&lr&dice during training or not')
    parser.add_argument('-use_custom_hyper', type=bool, default=False, help='Use user specified training hyperparameters')
    parser.add_argument('-lr_scheduler', default='constant', help='Learning rate scheduler type')
    args = parser.parse_args()
    
    opt = get_task_config(args.task)  # please configure your hyper-parameter
    opt.save_path_code = "_"
    opt.eval_mode = "dam_slice" 

    device = torch.device(opt.device)
    if args.keep_log:
        logtimestr = time.strftime('%m%d%H%M')  # initialize the tensorboard for record the training process
        boardpath = opt.tensorboard_path + args.modelname + opt.save_path_code + logtimestr
        if not os.path.isdir(boardpath):
            os.makedirs(boardpath)
        TensorWriter = SummaryWriter(boardpath)

    #  ============================= add the seed to make sure the results are reproducible ============================

    seed_value = 666  # the number of seed
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
    if args.use_custom_hyper:
        batch_size = opt.batch_size
        num_workers = opt.num_workers
        epochs = opt.epochs
        optimizer = optim.Adam(model.parameters(), lr=opt.base_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        lr_scheduler_type = args.lr_scheduler
    else:
        batch_size = mconfig.batch_size
        num_workers = mconfig.num_workers
        epochs = mconfig.epochs
        optimizer = model_optimizer
        lr_scheduler_type = mconfig.lr_scheduler
    
    if args.modelname == "DAM_GMIRS" or args.modelname == "DAM_MosMed" or args.modelname == "DAM":
        from models.DAM.data_dam import JointTransform2D, ImageToImage2D
    else:
        from utils.data_gmirs import JointTransform2D, ImageToImage2D
    
    if args.task == "MosMed" or args.task == "QaTaCovid19":
        from utils.data_SimpleTokenize import LViTGenerator as JointTransform2D

    tf_train = JointTransform2D(img_size=opt.img_size, crop=opt.crop, p_rota=0.2, p_scale=0.5, p_gaussn=0.0,
                                p_contr=0.5, p_gama=0.5, p_distor=0.0, color_jitter_params=None, long_mask=True)  # image reprocessing
    tf_val = JointTransform2D(img_size=opt.img_size, crop=opt.crop, color_jitter_params=None, long_mask=True)
    train_dataset = ImageToImage2D(opt.data_path, opt.train_split, tf_train, img_size=opt.img_size, max_tokens=args.max_tokens)
    val_dataset = ImageToImage2D(opt.data_path, opt.val_split, tf_val, img_size=opt.img_size, max_tokens=args.max_tokens)  # return image, mask, and filename
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    if lr_scheduler_type == "constant": 
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x: 1.0)
    elif lr_scheduler_type == "polynomial":
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x: (1 - x / (len(trainloader) * epochs)) ** 0.9)
    elif lr_scheduler_type == "cosineLR":
        lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-5)
    elif lr_scheduler_type == "plateau":
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
    elif lr_scheduler_type == "exp":
        lambda1 = lambda epoch: max(0.99**epoch, 0.1)
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda1)
    else:
        raise RuntimeError("Could not find this type learning rate scheduler:", lr_scheduler_type)

    model.to(device)
    if opt.pre_trained:
        model.load_state_dict(torch.load(opt.load_path))
    
    if args.n_gpu > 1:
        #model = nn.DataParallel(model, device_ids = [1,2,3])
        model = nn.DataParallel(model)
   
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total_params: {}".format(pytorch_total_params))

    #  ========================================== begin to train the model =============================================
    best_dice, best_epoch, loss_log, dice_log = 0.0, 0, np.zeros(epochs+1), np.zeros(epochs+1)
    for epoch in range(epochs):
        #  ------------------------------------ training ------------------------------------
        train_losses = 0
        for batch_idx, (datapack) in enumerate(trainloader):
            model.train()
            # ---------------------------------- forward ----------------------------------
            model_output = model(datapack, device)
            train_loss = model_output['loss']
            # ---------------------------------- backward ---------------------------------
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            lr_scheduler.step()

            train_losses += train_loss.item()

        #  ---------------------------- log the train progress ----------------------------
        print('epoch [{}/{}], train loss:{:.4f}'.format(epoch, opt.epochs, train_losses / (batch_idx + 1)))
        if args.keep_log:
            TensorWriter.add_scalar('train_loss', train_losses / (batch_idx + 1), epoch)
            TensorWriter.add_scalar('learning rate', optimizer.state_dict()['param_groups'][0]['lr'], epoch)
            loss_log[epoch] = train_losses / (batch_idx + 1)
            with open(opt.tensorboard_path + args.modelname + opt.save_path_code + logtimestr + '/trainloss.txt', 'w') as f:
                for i in range(len(loss_log)):
                    f.write(str(loss_log[i])+'\n')

        #  ----------------------------------- evaluate -----------------------------------
        if epoch % opt.eval_freq == 0:
            model.eval()
            val_output = get_eval(valloader, model, opt=opt, args=args)
            print('epoch [{}/{}], val loss:{:.4f}, val dice:{:.4f}'.format(epoch, opt.epochs, val_output['loss'], val_output['mdice']))
            if args.keep_log:
                TensorWriter.add_scalar('val_loss', val_output['loss'], epoch)
                TensorWriter.add_scalar('dices', val_output['mdice'], epoch)
                dice_log[epoch] = val_output['mdice']
                with open(opt.tensorboard_path + args.modelname + opt.save_path_code + logtimestr + '/dice.txt', 'w') as f:
                    for i in range(len(dice_log)):
                        f.write(str(dice_log[i])+'\n')
            if val_output['mdice'] > best_dice:
                best_dice = val_output['mdice']
                best_epoch = epoch
                timestr = time.strftime('%m%d%H%M')
                if not os.path.isdir(opt.save_path):
                    os.makedirs(opt.save_path)
                save_path = opt.save_path + args.modelname + opt.save_path_code + '%s' % timestr + '_' + str(epoch) + '_' + str(best_dice)
                torch.save(model.state_dict(), save_path + ".pth", _use_new_zipfile_serialization=False)

        if mconfig.early_stopping_patience>0 and (epoch - best_epoch) > mconfig.early_stopping_patience:
            save_path = opt.save_path + args.modelname + opt.save_path_code + '%s' % timestr + '_estop_' + str(epoch) + '_' + str(best_dice)
            torch.save(model.state_dict(), save_path + ".pth", _use_new_zipfile_serialization=False)
            break

        if epoch % opt.save_freq == 0 or epoch == (opt.epochs-1):
            if not os.path.isdir(opt.save_path):
                os.makedirs(opt.save_path)
            save_path = opt.save_path + args.modelname + opt.save_path_code + '_' + str(epoch)
            torch.save(model.state_dict(), save_path + ".pth", _use_new_zipfile_serialization=False)

if __name__ == '__main__':
    main()