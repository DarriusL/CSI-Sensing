# @Time   : 2023.09.22
# @Author : Darrius Lei
# @Email  : darrius.lei@outlook.com
from lib import glb_var, callback

logger = glb_var.get_value('logger');

def generate_model(net_cfg):
    from model.framework.mlp import MLP, MixMLP
    from model.framework.tsm import TSM
    device = glb_var.get_value('device');
    if net_cfg['name'] in ['AmplitudeMLP', 'PhaseMLP']:
        net =  MLP(net_cfg);
    elif net_cfg['name'] == 'MixMLP':
        net = MixMLP(net_cfg);
    elif  net_cfg['name'] in ['AmplitudeTSM', 'PhaseTSM']:
        net = TSM(net_cfg)
    else:
        logger.error(f'Name of net [{net_cfg["name"]}] is not supported.\nPlease replace or add by yourself.')
        raise callback.CustomException('NetCfgNameError');
    return net.to(device);