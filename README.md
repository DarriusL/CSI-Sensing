# CSI-Sensing

Code for 躺平小分队 in [第一届Wi-Fi感知大赛](https://competition.huaweicloud.com/information/1000041958/introduction)



## Command

### usage

```shell
 executor.py [-h] [--data_process_cfg DATA_PROCESS_CFG] [--config CONFIG] [--saved_config SAVED_CONFIG] [--mode MODE]
```

### options

```shell
  -h, --help            show this help message and exit
  --data_process_cfg DATA_PROCESS_CFG, -dp_cfg DATA_PROCESS_CFG
                        Config to process data
  --config CONFIG, -cfg CONFIG
                        config for run
  --saved_config SAVED_CONFIG, -sc SAVED_CONFIG
                        path for saved config to test
  --mode MODE           train/test/train_and_test
```

### qiuck start

data processing

```shell
python executor.py -dp_cfg='./config/data_process_cfg/all_with_label.json'
python executor.py -dp_cfg='./config/data_process_cfg/round_preliminary.json'
```

```shell
python executor.py -cfg='./config/mlp/amplitude_mlp.json' --mode='train'
python executor.py -cfg='./config/mlp/phase_mlp.json' --mode='train'
python executor.py -cfg='./config/mlp/mix_mlp.json' --mode='train'
```

```shell
python executor.py -cfg='./config/tsm/amplitude_tsm.json' --mode='train'
python executor.py -cfg='./config/tsm/amplitude_seq_tsm.json' --mode='train'
python executor.py -cfg='./config/tsm/phase_tsm.json' --mode='train'
```

