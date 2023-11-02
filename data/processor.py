# @Time   : 2023.09.21
# @Author : Darrius Lei
# @Email  : darrius.lei@outlook.com
from dataclasses import dataclass
from lib import glb_var, util, decorator
import numpy as np
import torch, os, time, scipy

logger = glb_var.get_value('logger');
packet_div_t = glb_var.get_value('packet_div_t');

@dataclass
class Data():
    '''
    Struct for csi data.

    data format:
    ------------
    t_stamps: [t] #remove already
    reals: [P, M, N, C, Ts]
    imags: [P, M, N, C, Ts]
    amplitudes: [P, M, N, C, Ts]
    phases: [P, M, N, C, Ts]
    labels: [P, L]

    Notes:
    ------
    t: Number of sampling points in the data set
    P: Number of packets
    M: Number of transmitter antennas
    N: Number of receiver antennas
    C: Number of channals
    Ts: Samples of each packets
    '''
    reals: None;
    imags: None;
    amplitudes: None;
    phases: None;
    labels: None;

def cal_amplitude(c):
    '''Calculate the amplitude for the complex
    '''
    return torch.sqrt(c.reals ** 2 + c.imags ** 2);

def cal_phase(c, eps = 1e-40):
    '''Calculate the phase for the complex
    '''
    return torch.atan(c.imags / ( c.reals + eps))

def t_str2float(t_str):
    '''Convert sting time stamp to float.
    '''
    return 3600 * int(t_str[0]) + 60 * int(t_str[1]) + float(t_str[2]);

def match_label(t_str, labels):
    '''Match label for each sting time stamp

    Parameters:
    -----------
    t_str: list or array
        sting time stamp
    
    labels: array 
        origin labels, one covers two seconds
    '''
    t = t_str2float(t_str);
    seq_t = 2 * np.arange(1, len(labels) + 1);
    return labels[seq_t >= t][0];

def packet_mask(t_stamps, p):
    return torch.bitwise_and(p*packet_div_t <=t_stamps, t_stamps <= (p+1)*packet_div_t);

def data_ext(cfg, packet_length):
    '''Extract data
    '''
    logger.info(f'Extracting dataset: {cfg["src"]}');
    t_start = time.time();
    #[P, M, N, C, Ts]
    shape = [-1, cfg['M'], cfg['N'], cfg['C'], packet_length];
    data = Data(None, None, None, None, None);
    data_lines = open(cfg['src'], 'r').readlines();
    if len(data_lines) == 1:
        #There is no line break between each sampling data in the second round of data set
        data_lines = np.array(data_lines[0].replace('i', 'j').split()).reshape(-1, np.prod(shape[1:4]) + cfg['delete_length'])
        #There is no need to separate by spaces
        is_line_split = True;
    else:
        is_line_split = False;
    
    if not is_line_split:
        t_truth = t_str2float(data_lines[-1].split()[:3]);
    else:
        t_truth = t_str2float(data_lines[-1, :3]);
    
    if cfg['label_cfg']['srcs'] is not None:
        labels_scd_dim = len(cfg['label_cfg']['srcs']);
        for lab_idx in range(labels_scd_dim):
            _, extension = os.path.splitext(cfg['label_cfg']['srcs'][lab_idx]);
            if extension == '.txt':
                labels = np.loadtxt(cfg['label_cfg']['srcs'][lab_idx]);
            elif extension == '.mat':
                labels = scipy.io.loadmat(cfg['label_cfg']['srcs'][lab_idx])['truth'].squeeze();
            t_label = 2 * len(labels);
            if not lab_idx:
                #[P, L]
                all_labels = np.zeros((len(labels), labels_scd_dim))
            all_labels[:, lab_idx] = labels;
        data.labels = torch.from_numpy(all_labels).to(torch.int64);
    else:
        labels_scd_dim = 1;
        t_label = t_truth;
        data.labels = torch.ones((int(np.ceil(t_truth/2)), labels_scd_dim), dtype=torch.int64) * cfg['label_cfg']['all_label_to'];

    shape[0] = data.labels.shape[0];
    shape = tuple(shape);
    data.reals = torch.zeros(shape);
    data.imags = torch.zeros_like(data.reals);
    data.amplitudes = torch.zeros_like(data.reals);
    data.phases = torch.zeros_like(data.reals);
    data.labels = torch.zeros((shape[0], labels_scd_dim), dtype = torch.int64);

    t_stamps = torch.zeros((len(data_lines)));
    #[M, N, C, t]
    reals = torch.zeros(shape[1:4] + (t_stamps.shape[0],))
    imags = torch.zeros_like(reals);

    for t, line in enumerate(data_lines):
        if not is_line_split:
            complex_strs = line.split();
        else:
            complex_strs = line.tolist();

        t_stamps[t] = t_str2float(complex_strs[:3]);
        
        del complex_strs[:cfg['delete_length']]

        for idx, c in enumerate(complex_strs):
            h_complex = complex(c);
            reals[idx//shape[3]//shape[2], idx//shape[3]%shape[2], idx%shape[3], t] = h_complex.real;
            imags[idx//shape[3]//shape[2], idx//shape[3]%shape[2], idx%shape[3], t] = h_complex.imag;

    t_idxs = torch.arange(t+1);
    #match packet
    for p in range(shape[0]):
        p_t_idxs = t_idxs[packet_mask(t_stamps, p)];
        if len(p_t_idxs) < packet_length:
            logger.error(f'{len(p_t_idxs)}')
            raise RuntimeError
        idxs = np.random.choice(len(p_t_idxs), packet_length);
        idxs.sort();
        p_t_idx = p_t_idxs[idxs];
        data.reals[p, :] = reals[:, :, :, p_t_idx];
        data.imags[p, :] = imags[:, :, :, p_t_idx];

    data.amplitudes = cal_amplitude(data);
    data.phases = cal_phase(data);
    logger.info(f'====================Info====================\n'
    f'dataset: {cfg["src"]}\n'
    f'length: {shape[-1]}\n'
    f'truth time/label time: {t_truth:.1f} s/{t_label:.1f} s\n'
    f'Saved directory: {cfg["tgt"]}\n'
    f'time consumption: {util.s2hms(time.time() - t_start)}')
    return data

def simplify_data(src):
    ''''''
    tgt = Data(None, None, None, None, None);
    tgt.amplitudes = src.amplitudes;
    tgt.phases = src.phases;
    tgt.labels = src.labels;
    return tgt


def divide(data, rate):
    '''Divide the data into a training set and a validation set according to a given ratio.
    Hint:This competition test should be data with continuous timestamps. 
        To ensure generality, a data set is randomly selected and a section is manually cut off as a test set.
    
    Parameters:
    -----------
    data: Data
        Extracted data

    rate: list
        train:valid
    
    Notes:
    -----------
    This is an abandoned solution. According to the data division by time slot, 
    the data in each time period (2s) is very similar, so a separate small data set needs to be used as verification.
    This code is temporarily retained to accommodate the possibility that packet data may be used later.
    '''
    assert sum(rate) <= 1
    n = len(data.t_stamps);
    n_train = int(np.ceil(n * rate[0]));
    train_idxs = np.random.choice(n, n_train, replace = False);
    valid_idxs = np.array(list(set(range(n)) - set(train_idxs)));
    train_idxs.sort(), valid_idxs.sort()
    idxs = [train_idxs, valid_idxs];

    datasets = {'train': None, 'valid': None};
    datas = [];
    for i in range(2):
        data_ = Data(None, None, None, None, None);
        data_.amplitudes = data.amplitudes[idxs[i], :, :];
        data_.phases = data.phases[idxs[i], :, :];
        data_.labels = data.labels[idxs[i]];
        datas.append(data_);
    datasets['train'] = datas[0];
    datasets['valid'] = datas[1];
    return datasets;

@decorator.Timer
def run_pcr(dp_cfg):
    packet_length = dp_cfg['packet_length'];
    for cfg in dp_cfg['datasets'].values():
        path, _ = os.path.split(cfg['tgt']);
        if not os.path.exists(path):
            os.makedirs(path);
        data = data_ext(cfg, packet_length);
        data = simplify_data(data);
        torch.save(data, cfg['tgt']);
