# @Time   : 2023.09.22
# @Author : Darrius Lei
# @Email  : darrius.lei@outlook.com

import torch
from lib import glb_var
from model import net_util
from model.framework.base import Net
#TODO: Multiple output
device = glb_var.get_value('device')

class MLP(Net):
    '''
    '''
    def __init__(self, net_cfg) -> None:
        super().__init__(net_cfg)
        M, N, C, Ts = self.input_dim[1:];
        activation_fn = net_util.get_activation_fn(self.activation_fn);
        self.loss_fn = net_util.get_loss(net_cfg['loss_cfg']);
        self.net = net_util.get_mlp_net(
            self.hid_layers, 
            activation_fn, 
            in_dim = M*N*C*Ts, 
            out_dim = self.category);
        if self.net_init:
            self.apply(self._init_para);
    
    def forward(self, data):
        '''
        data: list
        in:[batch_size, M, N, C, Ts]
        out:[batch_size, category]
        '''
        return self.net(data[0].flatten(1, -1));

    def cal_loss(self, logtis, labels):
        return self.loss_fn(logtis, labels);

class MixMLP(Net):
    '''
                 _ _ _ _ _ _                                          
               |           |                                      
    [input1]--| InPutNet1 |------ |    _ _ _ _ _ _ _               
             |_ _ _ _ _ _|       |    |             |              
                 ....           |----|  BodyNet    | ---[output]   
                _ _ _ _ _ __   |    |_ _ _ _ _ _ _|                
               |           |  |                                    
    [inputn]--| InPutNetn |--|                                    
             |_ _ _ _ _ _|                                              
    '''
    def __init__(self, net_cfg) -> None:
        super().__init__(net_cfg);
        activation_fn = net_util.get_activation_fn(self.activation_fn);
        self.loss_fn = net_util.get_loss(net_cfg['loss_cfg']);
        self.input_nets = torch.nn.ModuleList([net_util.get_mlp_net(
            self.input_hid_layers, 
            activation_fn, 
            in_dim = 4*64, 
            out_dim = self.input_out_dim)
            for _ in range(self.num_net)])
        self.body_net = net_util.get_mlp_net(
            self.body_hid_layers, 
            activation_fn, 
            in_dim = self.input_out_dim * self.num_net, 
            out_dim = self.category);
        if self.net_init:
            self.apply(self._init_para);

    def forward(self, data):
        '''
        data: list
        in:[batch_size, 4, 64]
        out:[batch_size, category]
        '''
        x = torch.zeros((data[0].shape[0], 0), device = device)
        for i, net in enumerate(self.input_nets):
            x = torch.cat(
                (x, net(data[i].flatten(1, -1))),
                dim = -1
            );
        return self.body_net(x);

    def cal_loss(self, logtis, labels):
        return self.loss_fn(logtis, labels);
        