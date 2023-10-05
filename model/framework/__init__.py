# @Time   : 2023.09.22
# @Author : Darrius Lei
# @Email  : darrius.lei@outlook.com
from lib import glb_var, callback

logger = glb_var.get_value('logger');

def generate_model(net_cfg):
    from model.framework.mlp import MLP, MixMLP
    device = glb_var.get_value('device');
    if net_cfg['type'] in ['AmplitudeMLP', 'PhaseMLP']:
        net =  MLP(net_cfg);
    elif net_cfg['type'] == 'MixMLP':
        net = MixMLP(net_cfg);
    else:
        logger.error(f'Type of net [{net_cfg["name"]}] is not supported.\nPlease replace or add by yourself.')
        raise callback.CustomException('NetCfgTypeError');
    return net.to(device);