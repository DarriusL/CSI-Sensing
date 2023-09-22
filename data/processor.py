# @Time   : 2023.09.21
# @Author : Darrius Lei
# @Email  : darrius.lei@outlook.com
from dataclasses import dataclass
from lib import glb_var, util
import numpy as np
import torch, os, time

logger = glb_var.get_value('logger');

#data format: [num_antenna, num_h, ts]
@dataclass
class Data():
    reals: None;
    imags: None;
    amplitudes: None;
    phases: None;
    labels: None

def cal_amplitude(c):
    '''
    '''
    return torch.sqrt(c.reals ** 2 + c.imags ** 2);

def cal_phase(c, eps = 1e-50):
    return torch.atan(c.imags / ( c.reals + eps))

def t_str2float(t_str):
    return 3600 * int(t_str[0]) + 60 * int(t_str[1]) + float(t_str[2]);

def match_label(t_str, labels):
    t = t_str2float(t_str);
    seq_t = 2 * np.arange(1, len(labels) + 1);
    return labels[seq_t >= t][0];

def data_ext(cfg):
    ''''''
    logger.info(f'Extracting dataset: {cfg["src"]}');
    t_start = time.time();
    data_lines = open(cfg['src'], 'r').readlines();
    t_truth = t_str2float(data_lines[-1].split()[:3])
    if cfg['label_cfg']['src'] is not None:
        labels = np.loadtxt(cfg['label_cfg']['src']).astype(np.int8);
        print( 'meta num of labels:', len(labels))
        set_labels = False;
        t_label = 2* len(labels);
    else:
        set_labels = True;
        t_label = t_truth;
    Ts = len(data_lines);
    data = Data(None, None, None, None, None);
    #data format: [num_antenna, num_h, ts]
    data.reals = torch.zeros((4, 64, Ts));
    data.imags = torch.zeros_like(data.reals);
    data.amplitudes = torch.zeros_like(data.reals);
    data.phases = torch.zeros_like(data.reals);
    data.labels = torch.zeros((Ts));

    for t, line in enumerate(data_lines):
        complex_strs = line.split()
        if not set_labels:
            data.labels[t] = match_label(complex_strs[:3], labels);
        else:
            data.labels[t] = cfg['label_cfg']['all_label_to'];
        del complex_strs[:12]
        for idx, c in enumerate(complex_strs):
            h_complex = complex(c);
            data.reals[idx//64, idx%64, t] = h_complex.real;
            data.imags[idx//64, idx%64, t] = h_complex.imag;

    data.amplitudes = cal_amplitude(data);
    data.phases = cal_phase(data);
    logger.info(f'====================Info====================\n'
    f'dataset: {cfg["src"]}\n'
    f'length: {Ts}\n'
    f'truth time/label time: {t_truth:.1f} s/{t_label:.1f} s\n'
    f'Saved directory: {cfg["tgt"]}'
    f'time consumption: {util.s2hms(time.time() - t_start)}')
    return data

def run_pcr(dp_cfg):
    for cfg in dp_cfg.values():
        data = data_ext(cfg)
        path, _ = os.path.split(cfg['tgt'])
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(data, cfg['tgt']);