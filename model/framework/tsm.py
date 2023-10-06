# @Time   : 2023.10.05
# @Author : Darrius Lei
# @Email  : darrius.lei@outlook.com

import torch
from lib import glb_var
from model import attnet, net_util
from model.framework.base import Net

class Encoder(torch.nn.Module):
    '''Encoder for EGPC

    Parameters:
    -----------
    d:int

    d_q:int
    Dimension of Q matrix

    d_k:int
    Dimension of K matrix

    d_v:int
    Dimension of V matrix

    d_fc:int
    Number of nodes in the dense network
    
    n_heads:int
    Number of the head

    posenc_buffer_size:int
    The length of the position embedding register

    is_norm_first:bool,optinal
    The sequence of attention network NORM
    '''
    def __init__(self, d, d_fc, n_heads, n_layers, 
                 posenc_buffer_size, is_norm_first = False) -> None:
        super().__init__();
        if is_norm_first:
            encoderlayer = attnet.EncoderLayer_PreLN;
        else:
            encoderlayer = attnet.EncoderLayer_PostLN;
        self.pos_embed = attnet.LearnablePositionEncoding(d, posenc_buffer_size);
        self.embed_dropout = torch.nn.Dropout(glb_var.get_value('dropout_rate'));
        #bad performance: self.pos_embed = attnet.PositionEncoding(d, max_len = posenc_buffer_size)
        self.Layers = torch.nn.ModuleList(
            [encoderlayer(d, d_fc, n_heads) for _ in range(n_layers)]
        )

    def forward(self, enc_input):
        #enc_input:(batch_size, antennas, d)
        #enc_output:(batch_size, antennas, d)
        #mask:(batch_size, antennas, antennas)

        enc_output = enc_input + self.pos_embed(enc_input);
        enc_output = self.embed_dropout(enc_output);
        '''
        o|x|x|x
        x|o|x|x
        x|x|o|x
        x|x|x|o
        '''
        mask = attnet.attn_subsequence_mask(enc_input);
        mask = torch.bitwise_or(mask, mask.transpose(-1, -2));
        for layer in self.Layers:
            enc_output = layer(enc_output, mask);
        #return:(batch_size, antennas, d)
        return enc_output;

class TSM(Net):
    '''
    '''
    def __init__(self, net_cfg) -> None:
        super().__init__(net_cfg)
        self.encoder = Encoder(64, self.d_fc, self.n_heads, self.n_layers, 
                               self.posenc_buffer_size, self.is_norm_first);
        self.loss_fn = net_util.get_loss(net_cfg['loss_cfg']);
        activation_fn = net_util.get_activation_fn(self.activation_fn);
        self.dense = net_util.get_mlp_net(
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
        return self.dense(self.encoder(data[0]).flatten(1, -1));

    def cal_loss(self, logtis, labels):
        return self.loss_fn(logtis, labels);
        
