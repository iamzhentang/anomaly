CUDA_VISIBLE_DEVICES=0 python3 main.py\
    --n_blocks=1\
    --batch_size=256\
    --window_size=7\
    --stride_size=1\
    --train_split=0.6\
    --name=Wadi\
    --data_dir='Dataset/input/total_daily_state_emission.csv'\
    > WADI.log 2>&1 &
#7表示每周为一个窗口，stride_size=1表示每次滑动一个窗口
