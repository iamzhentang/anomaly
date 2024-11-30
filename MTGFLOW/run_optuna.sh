CUDA_VISIBLE_DEVICES=0 python3 main_optuna.py\
    --window_size=1\
    --stride_size=1\
    --train_split=0.6\
    --name=Wadi\
    --data_dir='Dataset/input/t0.csv'\
    > WADI_optuna.log 2>&1 &

