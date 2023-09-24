# @Time   : 2023.09.22
# @Author : Darrius Lei
# @Email  : darrius.lei@outlook.com

import torch
from model import *
from room.officer import Trainer
from lib import json_util, glb_var, util

logger = glb_var.get_value('logger');

def run_work(cfg_path, mode):
    ''' Run work command

    Parameters:
    -----------
    config_path:str
    path of configure file(json file)

    mode:str, optional
    mode of work: train, test, train_and_test
    default:train
    '''
    #load config
    config = json_util.jsonload(cfg_path);
    
    #ser random seed
    util.set_seed(config['seed']);
    #initial device
    if config['gpu_is_available']:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu');
    else:
        device = torch.device('cpu');
    glb_var.set_value('device', device);

    #generate models based on configuration
    if mode in ['train', 'train_and_test']:
        model = generate_model(config['model'])
    else:
        pass

    if mode == 'train':
        trainer = Trainer(config, model);
        trainer.train();