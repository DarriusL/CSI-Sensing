# @Time   : 2023.09.22
# @Author : Darrius Lei
# @Email  : darrius.lei@outlook.com

from data import *
import numpy as np
import matplotlib.pyplot as plt
import torch, os
from lib import util, glb_var, callback, json_util, colortext, decorator

from model import *

logger = glb_var.get_value('logger');

class Trainer(object):
    '''Ordinary trainer
    '''
    def __init__(self, config, model) -> None:
        #note: config is global, not config['trian]
        cfg = config['train'];
        util.set_attr(self, cfg, except_type = dict);
        self.config = config;
        #Some simple initialization
        self.model = model;
        self.train_loss = [];
        self.train_acc = [];
        self.valid_loss = [];
        self.valid_acc = [];
        self.valid_min_acc = 0;
        self.optimizer = net_util.get_optimizer(cfg['optimizer_cfg'], self.model);
        self.lr_schedulr = net_util.get_lr_schedule(cfg['lr_schedule_cfg'], self.optimizer, self.max_epoch);
        self.train_wrapper, self.valid_wrapper = self._generate_loaderwrapper(cfg['dataset']);
        self.data_features = len(cfg['dataset']['loader_cfg']['csi_feature'])
        assert 1 <= self.data_features <= 2;
        self.save_path = './cache/saved/' + self.model.name + '/' + util.get_date() + '/' + util.get_time() + '/';
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path);
    
    def _load_dataset(self, srcs):
        if len(srcs) == 1:
            datasets = torch.load(srcs[0]);
        elif len(srcs) > 1:
            datasets = [torch.load(src) for src in srcs];
        return datasets

    def _generate_loaderwrapper(self, dataset_cfg):
        '''Generate wrapper for training and validation.
        '''
        dataset_train = self._load_dataset(dataset_cfg['src']);
        dataset_valid = self._load_dataset(dataset_cfg['valid_src'])
        train_wrapper = generate_LoaderWrapper(dataset_train, dataset_cfg['loader_cfg'], mode = 'train');
        valid_wrapper = generate_LoaderWrapper(dataset_valid, dataset_cfg['loader_cfg_valid'], mode = 'valid');
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
        self.optimizer.zero_grad();
        logits = self.model(csi_datas);
        loss = self.model.cal_loss(logits, labels);
        self._check_nan(loss);
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm = self.clip_grad_val);
        loss.backward();
        self.optimizer.step();
        return loss.item(), self._cal_acc(logits.argmax(dim = -1), labels);

    @torch.no_grad()
    def _valid_epoch(self, valid_data):
        '''Validation step for each epoch
        '''
        self.model.eval();
        *csi_datas, labels = valid_data;
        logits = self.model(csi_datas);
        return self.model.cal_loss(logits, labels).item(), self._cal_acc(logits.argmax(dim = -1), labels);

    def _save_point(self, epoch):
        info = {'epoch':epoch, 'train_acc':self.train_acc[-1], 'valid_acc':self.valid_acc[-1]};
        json_util.jsonsave(info, self.save_path + 'info.json');
        self.config['model_path'] = self.save_path + 'model.pt';
        json_util.jsonsave(self.config, self.save_path + 'config.json');
        torch.save(self.model, self.config['model_path']);

    @decorator.Timer
    def train(self):
        ''''''
        epoch_best, valid_not_improve_cnt = 0, 0;
        for epoch in range(self.max_epoch):
            losses = [];
            accs = [];
            for _ in range(self.train_times_per_epoch):
                #Why tuple:Iterative generators can only be used once, 
                # which is equivalent to storing them in temporary variables.
                batch = tuple(iter(self.train_wrapper).__next__());
                for _ in range(self.batch_learn_times_per_train):
                    loss, acc = self._train_epoch(batch);
                    losses.append(loss);
                    accs.append(acc);
            self.train_loss.append(np.mean(losses));
            self.train_acc.append(np.mean(accs));
            if self.show_train_info:
                logger.info(colortext.GREEN + f'[{self.model.name}] - [train]' + colortext.RESET + 
                            f'\n[epoch : {epoch + 1} / {self.max_epoch}] - lr: {self.optimizer.param_groups[0]["lr"]}'
                            f' - loss:{self.train_loss[-1]:.8f} - '
                            'acc(' + colortext.BLUE + 'now' + colortext.RESET +'/' + colortext.PURPLE + 'best' + 
                            colortext.RESET +'):' + colortext.BLUE +f'{self.train_acc[-1]:.8f}' + colortext.RESET + '/' + colortext.PURPLE + 
                            f'{max(self.train_acc):.8f}'+ colortext.RESET);
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
                if self.valid_acc[-1] > self.valid_min_acc or (self.valid_acc[-1] == self.valid_min_acc and self.train_acc[-1] == max(self.train_acc)):
                    self._save_point(epoch);
                    epoch_best = epoch;
                    self.valid_min_acc = self.valid_acc[-1];
                    valid_not_improve_cnt = 0;
                else:
                    valid_not_improve_cnt += 1;

                logger.info(colortext.RED +  f'[{self.model.name}] - [valid]' + colortext.RESET + 
                            f'\n[epoch : {epoch + 1} / {self.max_epoch}] - loss:{self.valid_loss[-1]:.8f}\n'
                            f'train acc(now/best):{self.train_acc[-1]:.8f}/{max(self.train_acc):.8f}\n'
                            f'valid acc(' + colortext.BLUE + 'now' + colortext.RESET +'/' + colortext.PURPLE + 'best' + 
                            colortext.RESET +'):' + colortext.BLUE +f'{self.valid_acc[-1]:.8f}' + colortext.RESET + '/' + colortext.PURPLE + 
                            f'{max(self.valid_acc):.8f}'+ colortext.RESET + 
                            '\n valid_not_improve_cnt:' + colortext.YELLOW + f'{valid_not_improve_cnt}' + colortext.RESET);
                if valid_not_improve_cnt >= self.stop_train_step_valid_not_improve:
                    logger.info('Meet the set requirements, stop training');
                    break;

        logger.info(f'[{self.model.name}]Training Summary:\n'
                    f'best (train/valid) acc : {max(self.train_acc):.8f}/{max(self.valid_acc):.8f}\n'
                    f'saving directory: {self.save_path}');

        plt.figure(figsize = (21, 6));
        plt.subplot(121)
        plt.plot(
            np.arange(0, len(self.train_loss)) + 1, 
            self.train_loss, 
            linewidth = 2, 
            label = 'train'
            );
        plt.plot(
            np.arange(self.valid_step - 1, len(self.train_loss), self.valid_step) + 1, 
            self.valid_loss, 
            linewidth=2,
            marker = 'o',
            label = 'valid'
            );
        plt.scatter(
            [epoch_best, epoch_best], 
            [self.train_loss[epoch_best], self.valid_loss[(epoch_best + 1)//self.valid_step - 1]], 
            c = 'red',
            marker = '^', 
            label = 'save point'
            );
        plt.xlabel('epoch');
        plt.ylabel('loss');
        plt.yscale('log');
        plt.legend(loc='upper right');
        plt.subplot(122)
        plt.plot(
            np.arange(0, len(self.train_acc)) + 1, 
            self.train_acc, 
            linewidth = 2, 
            label = 'train'
            );
        plt.plot(
            np.arange(self.valid_step - 1, len(self.train_acc), self.valid_step) + 1, 
            self.valid_acc, 
            linewidth=2,
            marker = 'o',
            label = 'valid'
            );
        plt.scatter(
            [epoch_best, epoch_best], 
            [self.train_acc[epoch_best], self.valid_acc[(epoch_best + 1)//self.valid_step - 1]], 
            c = 'red',
            marker = '^', 
            label = 'save point'
            );
        plt.xlabel('epoch');
        plt.ylabel('acc');
        plt.legend(loc='upper right');

        plt.savefig(self.save_path + '/output.png', dpi = 1100);





                
            