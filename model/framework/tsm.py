# @Time   : 2023.10.05
# @Author : Darrius Lei
# @Email  : darrius.lei@outlook.com

import torch
from lib import glb_var
from model import attnet, net_util
from model.framework.base import Net

device = glb_var.get_value('device')

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
        #enc_input:(batch_size, seq_len, d)
        #enc_output:(batch_size, seq_len, d)
        #mask:(batch_size, seq_len, seq_len)

        enc_output = enc_input + self.pos_embed(enc_input);
        enc_output = self.embed_dropout(enc_output);
        '''
        o|x|x|x
        x|o|x|x
        x|x|o|x
        x|x|x|o
        '''
        mask = attnet.attn_subsequence_mask(enc_input);
        #mask = torch.bitwise_or(mask, mask.transpose(-1, -2));
        for layer in self.Layers:
            enc_output = layer(enc_output, mask);
        #return:(batch_size, seq_len, d)
        return enc_output;

class TSM(Net):
    '''
    '''
    def __init__(self, net_cfg) -> None:
        super().__init__(net_cfg)
        self.batch_size, M, N, C, self.Ts = self.input_dim;
        self.encoder = Encoder(self.Ts, self.d_fc, self.n_heads, self.n_layers, 
                               self.posenc_buffer_size, self.is_norm_first);
        self.loss_fn = net_util.get_loss(net_cfg['loss_cfg']);
        activation_fn = net_util.get_activation_fn(self.activation_fn);
        self.denses = torch.nn.ModuleList([net_util.get_mlp_net(
            self.hid_layers, 
            activation_fn, 
            in_dim = M*N*C*self.Ts, 
            out_dim = self.category) for _ in range(self.n_outnets)]);
        if self.net_init:
            self.apply(self._init_para);
    
    def forward(self, data):
        '''
        data: list
        in:[batch_size, M, N, C, Ts]
        out:[batch_size, L, category]
        '''
        out = torch.zeros((self.batch_size, 0, self.category), device = device);
        feature = self.encoder(data[0].reshape(self.batch_size, -1, self.Ts)).flatten(1, -1);
        for dense in self.denses:
            out = torch.cat((out, dense(feature).unsqueeze(1)), dim = 1);
        return out;

    def cal_loss(self, logtis, labels):
        '''
        logits:[batch_size, L, category]
        labels:[batch_size, L]
        '''
        loss = torch.zeros((1), device = device);
        for l in range(self.n_outnets):
            loss += self.loss_fn(logtis[:, l, :], labels[:, l])
        return loss;
        
