# @Time   : 2023.09.22
# @Author : Darrius Lei
# @Email  : darrius.lei@outlook.com
from model.framework.mlp import MLP
from lib import glb_var, callback

logger = glb_var.get_value('logger');
device = glb_var.get_value('device');

def generate_model(net_cfg):
    if net_cfg['type'] in ['AmplitudeMLP']:
        net =  MLP(net_cfg);
    else:
        logger.error(f'Type of net [{net_cfg["name"]}] is not supported.\nPlease replace or add by yourself.')
        raise callback.CustomException('NetCfgTypeError');
    return net.to(device);