# @Time   : 2023.09.22
# @Author : Darrius Lei
# @Email  : darrius.lei@outlook.com

from data import *
import numpy as np
import matplotlib.pyplot as plt
import torch, os, time
from lib import util, glb_var, callback, json_util, colortext

from model import net_util

logger = glb_var.get_value('logger');

class Trainer():
    '''
    '''
    def __init__(self, cfg, model) -> None:
        #note: cfg is config['trian]
        util.set_attr(self, cfg, except_type = dict);
        #Some simple initialization
        self.model = model.to(glb_var.get_value('device')).train();
        self.train_loss = [];
        self.train_acc = [];
        self.valid_loss = [];
        self.valid_acc = [];
        self.valid_min_acc = 1;
        self.optimzer = net_util.get_optimizer(cfg['optimizer_cfg']);
        self.lr_schedulr = net_util.get_lr_schedule(cfg['lr_schedule_cfg'], self.optimzer, self.max_epoch);
        self.train_wrapper, self.valid_wrapper = self._generate_loaderwrapper(cfg['dataset']);
        self.data_features = len(cfg['dataset']['loader_cfg']['csi_feature'])
        assert 1 <= self.data_features <= 2;
    
    def _generate_loaderwrapper(dataset_cfg):
        '''Generate wrapper for training and validation.
        '''
        srcs = dataset_cfg['src'];
        if len(srcs) == 1:
            datasets = torch.load(srcs);
        elif len(srcs) > 1:
            datasets = [torch.load(src) for src in srcs];
        train_wrapper = generate_LoaderWrapper(datasets, dataset_cfg['loader_cfg'], mode = 'train');
        valid_wrapper = generate_LoaderWrapper(datasets, dataset_cfg['loader_cfg'], mode = 'valid');
        return train_wrapper, valid_wrapper
    
    def _check_nan(self, loss):
        if torch.isnan(loss):
            logger.error('Loss is nan.\nHint:\n(1)Check the loss function;\n'
                              '(2)Checks if the constant used is in the range between [-6.55 x 10^4]~[6.55 x 10^4]\n'
                              '(3)Not applicable for automatic mixed-precision acceleration.');
            raise callback.CustomException('ValueError');

    @torch.no_grad()
    def _cal_acc(self, labels_pred, labels):
        return (labels == labels_pred).float().mean().item();

    def _train_epoch(self, train_data):
        '''Training step for each epoch
        '''
        self.model.train();
        *csi_datas, labels = train_data;
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache();
        self.optimzer.zero_grad();
        logits = self.model(csi_datas);
        loss = self.model.cal_loss(logits, labels);
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm = self.clip_grad_val);
        self._check_nan(loss);
        loss.backward();
        self.optimzer.step();
        return loss.item(), self._cal_acc(logits.max(dim = -1).values);

    @torch.no_grad()
    def _valid_epoch(self, valid_data):
        '''Validation step for each epoch
        '''
        self.model.eval();
        *csi_datas, labels = valid_data;
        logits = self.model(csi_datas);
        return self.model.cal_loss(logits, labels).item(), self._cal_acc(logits.max(dim = -1).values);

    def _save_point(self):
        raise NotImplemented

    def train(self):
        t = time.time();
        valid_not_improve_cnt = 0;
        for epoch in range(self.max_epoch):
            loss, acc = self.model(iter(self.train_wrapper).__next__());
            self.train_loss.append(loss);
            self.train_acc.append(acc);
            logger.info(colortext.GREEN + f'[{self.model.type}] - [train]' + colortext.RESET + 
                        f'\n[epoch : {epoch + 1} / {self.max_epoch}] - lr: {self.optimizer.param_groups[0]["lr"]}'
                        f' - loss:{self.train_loss[-1]:.8f} - acc:{self.train_acc[-1]:.8f}');
            if self.lr_schedulr is not None:
                self.lr_schedulr.step();
            if (epoch + 1)%self.valid_step == 0:
                epoch_valid_loss, epoch_valid_acc = [], [];
                for _ in range(self.valid_times):
                    loss, acc = self._valid_epoch(iter(self.valid_wrapper).__next__());
                    epoch_valid_loss.append(loss);
                    epoch_valid_acc.append(acc);
                
                self.valid_loss.append(np.mean(epoch_valid_loss));
                self.valid_acc.append(np.mean(epoch_valid_acc));
                if self.valid_acc[-1] >= self.valid_min_acc:
                    self._save_point();
                    self.valid_min_acc = self.valid_acc[-1];
                    valid_not_improve_cnt = 0;
                else:
                    valid_not_improve_cnt += 1;

                logger.info(colortext.RED +  f'[{self.model.type}] - [valid]' + colortext.RESET + 
                            f'\n[epoch : {epoch + 1} / {self.max_epoch}] - loss:{self.valid_loss[-1]:.8f}'
                            f' - acc:{self.valid_acc[-1]:.8f} '
                            '\n valid_not_improve_cnt:' + colortext.YELLOW + f'{valid_not_improve_cnt}' + colortext.RESET);

#TODO:draw





                
            