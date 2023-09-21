# @Time   : 2023.09.21
# @Author : Darrius Lei
# @Email  : darrius.lei@outlook.com
from dataclasses import dataclass
import numpy as np
import torch, os

#data format: [num_antenna, num_h, ts]
@dataclass
class Data():
    real: None;
    imag: None;
    amplitude: None;
    phase: None;
    label: None

def cal_amplitude(c):
    '''
    '''
    return torch.sqrt(c.real ** 2 + c.imag ** 2);

def cal_phase(c, eps = 1e-50):
    return torch.atan(c.imag / ( c. real + eps))

def match_label(t_str, labels):
    t = 3600 * int(t_str[0]) + 60 * int(t_str[1]) + int(t_str[2]);
    seq_t = 2 * np.arange(1, len(labels) + 1);
    return labels[seq_t >= t][0];

def data_ext(cfg):
    ''''''
    data_lines = open(cfg['src'], 'r').readlines();
    if cfg['label_cfg']['src'] is not None:
        labels = np.loadtxt(cfg['label_cfg']['src']).astype(np.int8);
        print( 'meta num of labels:', len(labels))
        set_labels = False;
    else:
        set_labels = True;
    Ts = len(data_lines);
    print('ts:', Ts)
    data = Data(None, None, None, None, None);
    #data format: [num_antenna, num_h, ts]
    data.real = torch.zeros((4, 64, Ts));
    data.imag = torch.zeros_like(data.real);
    data.amplitude = torch.zeros_like(data.real);
    data.phase = torch.zeros_like(data.real);
    data.label = torch.zeros((Ts));

    for t, line in enumerate(data_lines):
        complex_strs = line.split()
        if not set_labels:
            data.label[t] = match_label(complex_strs[:3], labels);
        else:
            data.label[t] = cfg['label_cfg']['all_label_to'];
        del complex_strs[:12]
        for idx, c in enumerate(complex_strs):
            h_complex = complex(c);
            data.real[idx//64, idx%64, t] = h_complex.real;
            data.imag[idx//64, idx%64, t] = h_complex.imag;

    data.amplitude = cal_amplitude(data);
    data.phase = cal_phase(data);
    return data

def run_pcr(dp_cfg):
    for cfg in dp_cfg.values():
        data = data_ext(cfg)
        if not os.path.exists(cfg['tgt']):
            os.makedirs(cfg['tgt'])
        torch.save(data, cfg['tgt']);