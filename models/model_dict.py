
# This file is used to obtain the model
from models.config_model import get_model_config

def get_model(modelname, args, opt=None):
    model_config = get_model_config(modelname)
    if modelname == "DAM_GMIRS":
        import models.DAM.Config_GMIRS_DAM as config
        from models.DAM.backbone import DAM
        config_model = config.get_model_config()
        model = DAM(config_model)
        optimizer = model_config.get_optimizer(model, lr=model_config.lr, weight_decay=model_config.weight_decay)
    
    elif modelname == "DAM_MosMed":
        import models.DAM.Config_MosMed_DAM as config
        from models.DAM.backbone import DAM
        config_model = config.get_model_config()
        model = DAM(config_model)
        optimizer = model_config.get_optimizer(model, lr=model_config.lr, weight_decay=model_config.weight_decay)
    
    else:
        raise RuntimeError("Could not find the model:", modelname)
    return model, optimizer, model_config
