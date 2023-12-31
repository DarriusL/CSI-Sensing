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
    glb_var.set_values(json_util.jsonload('./config/constant_cfg.json'), except_type=dict);
    parse = argparse.ArgumentParser();
    parse.add_argument('--data_process_cfg', '-dp_cfg', type = str, default = None, help = 'Config to process data');
    parse.add_argument('--config', '-cfg', type = str, default = None, help = 'config for run');
    parse.add_argument('--saved_config', '-sc', type = str, default = None, help = 'path for saved config to test')
    parse.add_argument('--mode', type = str, default = 'train', help = 'train/test/train_and_test')

    args = parse.parse_args();

    #execute date process command
    if args.data_process_cfg is not None:
        from data.processor import run_pcr
        run_pcr(json_util.jsonload(args.data_process_cfg));
        sys.exit(0);

    #execute work command
    from room import run_work
    if args.config is not None:
        if args.mode not in ['train', 'train_and_test']:
            log.error(f'The using of the configuration file does not support the {args.mode} mode. \n'
                      'Prompt: first use the [train] mode and then use the command [-- saved_config] to run the [test] mode, \n'
                      'or directly use the [train_and_test] mode');
            raise CustomException('ModeError');
        run_work(args.config, args.mode);
    elif args.saved_config is not None:
        if args.mode in ['train', 'train_and_test']:
            log.error('The saved model file only supports test mode');
            raise CustomException('ModeError');
        run_work(args.saved_config, args.mode);


    