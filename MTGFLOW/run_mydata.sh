CUDA_VISIBLE_DEVICES=0 python3 main.py\
    --n_blocks=2\
    --batch_size=256\
    --window_size=1\
    --stride_size=1\
    --train_split=0.6\
    --name=Wadi\
    --data_dir='Dataset/input/total_daily_state_emission.csv'\
    > WADI.log 2>&1 &

