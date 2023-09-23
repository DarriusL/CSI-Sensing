# @Time   : 2023.09.21
# @Author : Darrius Lei
# @Email  : darrius.lei@outlook.com
from dataclasses import dataclass
from lib import glb_var, util
import numpy as np
import torch, os, time, scipy

logger = glb_var.get_value('logger');

@dataclass
class Data():
    '''
    Struct for csi data.

    data format:
    ------------
    t_stamps: [ts]
    reals: [ts, num_antenna, num_csi]
    imags: [ts, num_antenna, num_csi]
    amplitudes: [ts, num_antenna, num_csi]
    phases: [ts, num_antenna, num_csi]
    labels: [ts]
    '''
    #time stamp is only for test
    t_stamps: None;
    reals: None;
    imags: None;
    amplitudes: None;
    phases: None;
    labels: None;

def cal_amplitude(c):
    '''Calculate the amplitude for the complex
    '''
    return torch.sqrt(c.reals ** 2 + c.imags ** 2);

def cal_phase(c, eps = 1e-50):
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

def data_ext(cfg):
    '''Extract data
    '''
    logger.info(f'Extracting dataset: {cfg["src"]}');
    t_start = time.time();
    data_lines = open(cfg['src'], 'r').readlines();
    if len(data_lines) == 1:
        #There is no line break between each sampling data in the second round of data set
        data_lines = np.array(data_lines[0].replace('i', 'j').split()).reshape(-1, 256 + 12)
        #There is no need to separate by spaces
        is_line_split = True;
    else:
        is_line_split = False;
    
    if not is_line_split:
        t_truth = t_str2float(data_lines[-1].split()[:3]);
    else:
        t_truth = t_str2float(data_lines[-1, :3]);
    
    if cfg['label_cfg']['src'] is not None:
        _, extension = os.path.splitext(cfg['label_cfg']['src']);
        if extension == '.txt':
            labels = np.loadtxt(cfg['label_cfg']['src']).astype(np.int8);
        elif extension == '.mat':
            labels = scipy.io.loadmat(cfg['label_cfg']['src'])['truth'].squeeze();
        print( 'meta num of labels:', len(labels))
        set_labels = False;
        t_label = 2* len(labels);
    else:
        set_labels = True;
        t_label = t_truth;
    Ts = len(data_lines);
    data = Data(None, None, None, None, None, None);
    data.reals = torch.zeros((Ts, 4, 64));
    data.imags = torch.zeros_like(data.reals);
    data.amplitudes = torch.zeros_like(data.reals);
    data.phases = torch.zeros_like(data.reals);
    data.labels = torch.zeros((Ts), dtype = torch.int64);
    data.t_stamps = torch.zeros_like(data.reals);

    for t, line in enumerate(data_lines):
        if not is_line_split:
            complex_strs = line.split();
        else:
            complex_strs = line.tolist();
        
        data.t_stamps[t] = t_str2float(complex_strs[:3]);

        if not set_labels:
            data.labels[t] = match_label(complex_strs[:3], labels);
            data.truths = torch.from_numpy(labels).to(torch.int64);
        else:
            data.labels[t] = cfg['label_cfg']['all_label_to'];
        
        del complex_strs[:12]

        for idx, c in enumerate(complex_strs):
            h_complex = complex(c);
            data.reals[t, idx//64, idx%64] = h_complex.real;
            data.imags[t, idx//64, idx%64] = h_complex.imag;

    data.amplitudes = cal_amplitude(data);
    data.phases = cal_phase(data);
    logger.info(f'====================Info====================\n'
    f'dataset: {cfg["src"]}\n'
    f'length: {Ts}\n'
    f'truth time/label time: {t_truth:.1f} s/{t_label:.1f} s\n'
    f'Saved directory: {cfg["tgt"]}\n'
    f'time consumption: {util.s2hms(time.time() - t_start)}')
    return data

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
    '''
    assert sum(rate) <= 1
    n = len(data.t_stamps);
    n_train = int(np.ceil(n * rate[0]));
    train_idxs = np.random.choice(n, n_train, replace = False);
    valid_idxs = np.array(list(set(range(n)) - set(train_idxs)));
    idxs = [train_idxs, valid_idxs];

    datasets = {'train': None, 'valid': None};
    datas = [];
    for i in range(2):
        data_ = Data(None, None, None, None, None, None);
        #
        data_.amplitudes = data.amplitudes[idxs[i], :, :];
        data_.phases = data.phases[idxs[i], :, :];
        data_.labels = data.labels[idxs[i]];
        datas.append(data_);
    datasets['train'] = datas[0];
    datasets['valid'] = datas[1];
    return datasets;

def generate_test_data(data, cfg):
    ''''''
    data_ = Data(None, None, None, None, None, None);
    data_.t_stamps = data.t_stamps;
    data_.amplitudes = data.amplitudes;
    data_.phases = data.phases;
    #No need to match the label.
    data_.labels =  data.truths;
    return data_;

def run_pcr(dp_cfg):
    t = time.time();
    for cfg in dp_cfg['datasets'].values():
        path, _ = os.path.split(cfg['tgt']);
        if not os.path.exists(path):
            os.makedirs(path);
        data = data_ext(cfg);
        if 'test_tgt' in cfg.keys():
            test_data = generate_test_data(data, cfg);
            torch.save(test_data, cfg['test_tgt']);
        data = divide(data, dp_cfg['division_ratio']);
        torch.save(data, cfg['tgt']);
    logger.info(f'Total processing time: {util.s2hms(time.time() - t)}');