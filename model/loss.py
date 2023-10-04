# @Time   : 2023.09.24
# @Author : Darrius Lei
# @Email  : darrius.lei@outlook.com

import torch
from lib import glb_var, callback

logger = glb_var.get_value('logger')
device = glb_var.get_value('device')

class FocalLoss(torch.nn.Module):
    ''' Multi-category focal loss with weights

    Parameters:
    -----------
    alphas: one-dimensional iterator
        Weights for all class, where index is label.
    
    gamma: float, optional
        Regulating factor
        default:2

    reduction:str, optional('mean'/'sum')
        How to calculate the final output
        default:'mean'

    Config:
    -------
    "loss_cfg":{
        "name":"FocalLoss",
        "alphas":[
            0.1,
            0.01,
            ...
        ],
        "gamma":2,
        "reduction":"mean"
    }
    
    Reference:
    ----------
    [1]Lin T Y, Goyal P, Girshick R, et al. 
    Focal loss for dense object detection[C]//
    Proceedings of the IEEE international conference on computer vision. 2017: 2980-2988.

    [2]https://github.com/li199603/pytorch_focal_loss
    '''
    def __init__(self, alphas, gamma = 2, reduction = 'mean') -> None:
        super().__init__();
        self.alphas = torch.as_tensor(alphas, device = device);
        self.gamma = gamma;
        self.reduction = reduction;

    def forward(self, logtis, targets):
        '''
        Calculate focal loss

        Parameters:
        -----------

        logits: torch.Tensor
            Model raw output, [batch_size, n]

        targets: torch.Tensor
            Corresponding label, [batch_size]
        '''
        alphas = self.alphas[targets];
        #[batch_size]
        logpt = torch.log_softmax(logtis, dim = -1).gather(dim = - 1, index = targets.reshape(-1, 1)).squeeze(-1);
        ce_loss, pt = -logpt, torch.exp(logpt);
        #[batch_size]
        loss = alphas * (1 - pt) ** self.gamma * ce_loss;
        if self.reduction == 'mean':
            return loss.mean(dim = 0);
        elif self.reduction == 'sum':
            return loss.sum(dim = 0);
        else:
            logger.error(f'Unsupported reduction type {self.reduction}');
            raise callback.CustomException('CfgError');


