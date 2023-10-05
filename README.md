# CSI-Sensing

Code for 躺平小分队 in [第一届Wi-Fi感知大赛](https://competition.huaweicloud.com/information/1000041958/introduction)



## Command

### usage

```shell
executor.py [-h] [--data_process DATA_PROCESS] [--config CONFIG] [--saved_config SAVED_CONFIG] [--mode MODE]
```

### options

```shell
  -h, --help            show this help message and exit
  --data_process DATA_PROCESS, -dp DATA_PROCESS
                        Whether to process data(True/False)
  --config CONFIG, -cfg CONFIG
                        config for run
  --saved_config SAVED_CONFIG, -sc SAVED_CONFIG
                        path for saved config to test
  --mode MODE           train/test/train_and_test
```

### qiuck start

data processing

```shell
python executor.py -dp=True
python executor.py --data_process=True
```

```shell
python executor.py -cfg='./config/mlp/amplitude_mlp.json' --mode='train'
python executor.py -cfg='./config/mlp/phase_mlp.json' --mode='train'
python executor.py -cfg='./config/mlp/mix_mlp.json' --mode='train'
```

