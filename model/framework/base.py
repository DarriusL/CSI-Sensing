# @Time   : 2023.09.22
# @Author : Darrius Lei
# @Email  : darrius.lei@outlook.com

import torch
from lib import glb_var, util

logger = glb_var.get_value('logger');

class Net(torch.nn.Module):
    '''Abstract Net class to define the API methods
    '''
    def __init__(self, net_cfg) -> None:
        super().__init__();
        util.set_attr(self, net_cfg, except_type = dict);

    def _init_para(self, module):
        if isinstance(module, torch.nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=1/module.embedding_dim)
        elif isinstance(module, torch.nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, torch.nn.Linear):
            module.weight.data.normal_()
            if module.bias is not None:
                module.bias.data.zero_()
    
    def forward(self):
        '''network forward pass'''
        logger.error('Method needs to be called after being implemented');
        raise NotImplementedError;

    def cal_loss(self):
        '''Calculate the loss'''
        logger.error('Method needs to be called after being implemented');
        raise NotImplementedError;

    def classifier(self, data):
        '''General Classifier for all model'''
        return self.forward(data).argmax(dim = -1);