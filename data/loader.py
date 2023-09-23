# @Time   : 2023.09.21
# @Author : Darrius Lei
# @Email  : darrius.lei@outlook.com

import torch, platform
from lib import glb_var

logger = glb_var.get_value('logger');

class Dataset(torch.utils.data.Dataset):
    '''Abstract Dataset Class'''
    def __init__(self, length) -> None:
        super().__init__();
        self.length = length;

    def __len__(self):
        return self.length;

    def __getitem__(self):
        logger.error('Method needs to be called after being implemented');
        raise NotImplementedError;

class SingleDataset(Dataset):
    '''Dataset for single data

    Sampled data format:
    data0: [batch_size, 4, 64]
    (data1:[batch_size, 4, 64])
    labels:[batch_size]
    '''
    def __init__(self, data, features):
        super().__init__(len(data.labels));
        #The order is aligned with the order configured in features.
        #for item in datas: [t, 4, 64]
        self.datas = [getattr(data, feature) for feature in features];
        self.labels = data.labels;

    def __getitem__(self, index):
        return tuple([data[index, :, :] for data in self.datas] + [self.labels[index]]);

class MultiDataset(SingleDataset):
    '''Dataset for multiple data
    '''
    def __init__(self, datasets, features, mode):
        datas = [dataset[mode] for dataset in datasets];
        self.datas = [];
        self.labels = torch.zeros((0), dtype = torch.int64);
        for feature in features:
            data_ = torch.zeros((0, 4, 64));
            for data in datas:
                data_ = torch.cat((data_, getattr(data, feature)));
                if feature == features[0]:
                    self.labels = torch.cat((self.labels, data.labels), dim = 0);
            self.datas.append(data_);
        Dataset.__init__(self, length=len(self.labels));

class LoaderWrapper():
    '''Wrapper for loaer
    Reserved for later use, if the data needs to be processed later
    '''
    def __init__(self, loader) -> None:
        self.loader = loader;

    def __iter__(self):
        device = glb_var.get_value('device')
        for batch in self.loader:
            yield (data.to(device) for data in batch)
    
    def process(self):
        pass 


def generate_LoaderWrapper(datasets, loader_cfg, mode):
    '''Gnerate DataLoader

    Parameters:
    ----------
    datasets: dict or list
        When using SingleDataset, datasets(dict) is single.
    
    loader_cfg:dict

    mode: str
    '''
    num_workers = loader_cfg['linux_num_workers'] if platform.system().lower() == 'linux' else 0;
    if isinstance(datasets, dict):
        return LoaderWrapper(
            torch.utils.data.DataLoader(
                SingleDataset(datasets[mode], loader_cfg['csi_feature']),
                num_workers = num_workers,
                pin_memory = True,
                batch_size = loader_cfg['batch_size'],
                shuffle = loader_cfg['shuffle']
                )
            );
    elif isinstance(datasets, list):
        return LoaderWrapper(
            torch.utils.data.DataLoader(
                MultiDataset(datasets, loader_cfg['csi_feature'], mode),
                num_workers = num_workers,
                pin_memory = True,
                batch_size = loader_cfg['batch_size'],
                shuffle = loader_cfg['shuffle'] 
            )
        );
    else:
        logger.error(f'No suitable module for the input [datasets: {type(datasets)}]');
        raise ModuleNotFoundError