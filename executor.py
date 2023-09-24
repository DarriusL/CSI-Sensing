# @Time   : 2023.09.21
# @Author : Darrius Lei
# @Email  : darrius.lei@outlook.com

import argparse, sys, logging, os
from lib import glb_var, json_util
from lib.callback import Logger, CustomException

if __name__ == '__main__':
    if not os.path.exists('./cache/logger/'):
        os.makedirs('./cache/logger/')
    glb_var.__init__();
    log = Logger(
        level = logging.INFO,
        filename = './cache/logger/logger.log',
    ).get_log()
    glb_var.set_value('logger', log);
    parse = argparse.ArgumentParser();
    parse.add_argument('--data_process', '-dp', type = bool, default = False, help = 'Whether to process data(True/False)');
    parse.add_argument('--config', '-cfg', type = str, default = None, help = 'config for run');
    parse.add_argument('--saved_config', '-sc', type = str, default = None, help = 'path for saved config to test')
    parse.add_argument('--mode', type = str, default = 'train', help = 'train/test/train_and_test')

    args = parse.parse_args();

    #execute date process command
    if args.data_process:
        from data.processor import run_pcr
        run_pcr(json_util.jsonload('./config/data_process_cfg.json'));
        sys.exit(0);

    #execute work command
    from room import *

    