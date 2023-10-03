# @Time   : 2023.09.22
# @Author : Darrius Lei
# @Email  : darrius.lei@outlook.com

from model import net_util
from model.framework.base import Net
import torch

class MLP(Net):
    '''
    '''
    def __init__(self, net_cfg) -> None:
        super().__init__(net_cfg)
        activation_fn = net_util.get_activation_fn(self.activation_fn);
        self.loss_fn = net_util.get_loss(net_cfg['loss_cfg']);
        self.net = net_util.get_mlp_net(
            self.hid_layers, 
            activation_fn, 
            in_dim = 4*64, 
            out_dim = self.category);
        if self.net_init:
            self.apply(self._init_para);
    
    def forward(self, data):
        '''
        data: list
        in:[batch_size, 4, 64]
        out:[batch_size, category]
        '''
        return self.net(data[0].flatten(1, -1));

    def cal_loss(self, logtis, labels):
        return self.loss_fn(logtis, labels);
        