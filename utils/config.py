# This file is used to configure the training parameters for each task

class Config_INSTANCE:
    data_path = "/home/lx/dataset/CT/"
    save_path = "./checkpoints/INSTANCE/"
    result_path = "./result/INSTANCE/"
    tensorboard_path = "./tensorboard/INSTANCE/"
    load_path = save_path + "xxxx.pth"

    num_workers = 8                  # number of data loading workers (default: 8)
    epochs = 300                 # number of total epochs to run (default: 400)
    batch_size = 4               # batch size (default: 4)
    base_lr = 1e-4         # initial learning rate (default: 0.001)
    momentum = 0.9               # momentum
    classes = 2                 # the number of classes (background + foreground)
    img_size = 256               # the input size of model
    train_split = "train_instance"        # the file name of training set
    val_split = "val_instance"           # the file name of testing set
    test_split = "test_instance"
    crop = None                  # the cropped image size
    eval_freq = 1                # the frequency of evaluate the model
    save_freq = 2000               # the frequency of saving the model
    device = "cuda"              # training device, cpu or cuda
    cuda = "on"                  # switch on/off cuda option (default: off)
    gray = "yes"                 # the type of input image
    img_channel = 1              # the channel of input image
    eval_mode = "slice"     # the mode when evaluate the model, slice level or patient level
    pre_trained = False
    mode = "train"
    visual = False
    modelname = "GMIRS"


class Config_MosMed:
    data_path = "/home/lx/dataset/GMIRS/"
    save_path = "./checkpoints/MosMed/"
    result_path = "./result/MosMed/"
    tensorboard_path = "./tensorboard/MosMed/"
    load_path = save_path + "/xxx.pth"

    num_workers = 8                  # number of data loading workers (default: 8)
    epochs = 300                 # number of total epochs to run (default: 400)
    batch_size = 8               # batch size (default: 4)
    base_lr = 1e-4         # initial learning rate (default: 0.001)
    momentum = 0.9               # momentum
    classes = 2                 # the number of classes (background + foreground)
    img_size = 224               # the input size of model
    train_split = "train_mosmeddataplus"        # the file name of training set
    val_split = "val_mosmeddataplus"           # the file name of testing set
    test_split = "test_mosmeddataplus"
    crop = None                  # the cropped image size
    eval_freq = 1                # the frequency of evaluate the model
    save_freq = 2000               # the frequency of saving the model
    device = "cuda"              # training device, cpu or cuda
    cuda = "on"                  # switch on/off cuda option (default: off)
    gray = "yes"                 # the type of input image
    img_channel = 1              # the channel of input image
    eval_mode = "slice"     # the mode when evaluate the model, slice level or patient level
    pre_trained = False
    mode = "train"
    visual = False
    modelname = "GMIRS"

class Config_QaTaCovid19:
    data_path = "/home/lx/dataset/GMIRS/"
    save_path = "./checkpoints/QaTaCovid19/"
    result_path = "./result/QaTaCovid19/"
    tensorboard_path = "./tensorboard/QaTaCovid19/"
    load_path = save_path + "/xxx.pth"

    num_workers = 8                  # number of data loading workers (default: 8)
    epochs = 300                 # number of total epochs to run (default: 400)
    batch_size = 8               # batch size (default: 4)
    base_lr = 1e-4         # initial learning rate (default: 0.001)
    momentum = 0.9               # momentum
    classes = 2                 # the number of classes (background + foreground)
    img_size = 224               # the input size of model
    train_split = "train_qatacovid19"        # the file name of training set
    val_split = "val_qatacovid19"           # the file name of testing set
    test_split = "test_qatacovid19"
    crop = None                  # the cropped image size
    eval_freq = 1                # the frequency of evaluate the model
    save_freq = 2000               # the frequency of saving the model
    device = "cuda"              # training device, cpu or cuda
    cuda = "on"                  # switch on/off cuda option (default: off)
    gray = "yes"                 # the type of input image
    img_channel = 1              # the channel of input image
    eval_mode = "slice"     # the mode when evaluate the model, slice level or patient level
    pre_trained = False
    mode = "train"
    visual = False
    modelname = "GMIRS"


class Config_GMIRS:
    data_path = "/home/lx/dataset/GMIRS/"
    save_path = "./checkpoints/GMIRS/"
    result_path = "./result/GMIRS/"
    tensorboard_path = "./tensorboard/GMIRS/"
    load_path = save_path + "/xxx.pth"

    num_workers = 8                  # number of data loading workers (default: 8)
    epochs = 200                 # number of total epochs to run (default: 400)
    batch_size = 4               # batch size (default: 4)
    base_lr = 1e-4         # initial learning rate (default: 0.001)
    momentum = 0.9               # momentum
    classes = 2                 # the number of classes (background + foreground)
    img_size = 256               # the input size of model
    train_split ="train_gmirs_w_text_label"   # the file name of training set
    val_split = "val_gmirs_w_text_label" # the file name of testing set
    test_split = "test_gmirs_w_text_label"  # the file name of testing set
    crop = None                  # the cropped image size
    eval_freq = 1                # the frequency of evaluate the model
    save_freq = 2000               # the frequency of saving the model
    device = "cuda"              # training device, cpu or cuda
    cuda = "on"                  # switch on/off cuda option (default: off)
    gray = "yes"                 # the type of input image
    img_channel = 1              # the channel of input image
    eval_mode = "combined_slice"     # the mode when evaluate the model, slice level or patient level
    pre_trained = False
    mode = "train"
    visual = False
    modelname = "GMIRS"


class Config_WORD:
    data_path = "/home/lx/dataset/CT/"
    save_path = "./checkpoints/WORD/"
    result_path = "./result/WORD/"
    tensorboard_path = "./tensorboard/WORD/"
    load_path = save_path + "/xxx.pth"

    num_workers = 8                  # number of data loading workers (default: 8)
    epochs = 400                 # number of total epochs to run (default: 400)
    batch_size = 8              # batch size (default: 4)
    base_lr = 1e-4         # initial learning rate (default: 0.001)
    momentum = 0.9               # momentum
    classes = 2                 # the number of classes (background + foreground)
    img_size = 256               # the input size of model
    train_split = "train_word"        # the file name of training set
    val_split = "val_word"           # the file name of testing set
    test_split = "test_word"
    crop = None                  # the cropped image size
    eval_freq = 1                # the frequency of evaluate the model
    save_freq = 2000               # the frequency of saving the model
    device = "cuda"              # training device, cpu or cuda
    cuda = "on"                  # switch on/off cuda option (default: off)
    gray = "yes"                 # the type of input image
    img_channel = 1              # the channel of input image
    eval_mode = "slice"     # the mode when evaluate the model, slice level or patient level
    pre_trained = False
    mode = "train"
    visual = False
    modelname = "GMIRS"

def get_task_config(task="GMIRS"):
    if task == "INSTANCE":
        return Config_INSTANCE()
    elif task == "MosMed":
        return Config_MosMed()
    elif task == "QaTaCovid19":
        return Config_QaTaCovid19()
    elif task == "GMIRS":
        return Config_GMIRS()
    elif task == "WORD":
        return Config_WORD()
    else:
        assert("We have not configured this task, please configure it first or use the configured task.")