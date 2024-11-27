#%%
import os
import argparse
import torch
from models.MTGFLOW import MTGFLOW
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve 
import pandas as pd

parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', type=str, 
                    default='Data/input/SWaT_Dataset_Attack_v0.csv', help='Location of datasets.')
parser.add_argument('--output_dir', type=str, 
                    default='./checkpoint/')
parser.add_argument('--name',default='SWaT', help='the name of dataset')

parser.add_argument('--graph', type=str, default='None')
parser.add_argument('--model', type=str, default='MAF')


parser.add_argument('--n_blocks', type=int, default=1, help='Number of blocks to stack in a model (MADE in MAF; Coupling+BN in RealNVP).')
parser.add_argument('--n_components', type=int, default=1, help='Number of Gaussian clusters for mixture of gaussians models.')
parser.add_argument('--hidden_size', type=int, default=32, help='Hidden layer size for MADE (and each MADE block in an MAF).')
parser.add_argument('--n_hidden', type=int, default=1, help='Number of hidden layers in each MADE.')
parser.add_argument('--input_size', type=int, default=1)
parser.add_argument('--batch_norm', type=bool, default=False)
parser.add_argument('--train_split', type=float, default=0.6)
parser.add_argument('--stride_size', type=int, default=10)

parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--window_size', type=int, default=60)
parser.add_argument('--lr', type=float, default=2e-3, help='Learning rate.')



args = parser.parse_known_args()[0]
args.cuda = torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")

def save_png(n_by_one_df):
    # 创建x轴的序号
    x = np.arange(n_by_one_df.shape[1])

    # 获取y轴的值（取绝对值）
    y = np.abs(n_by_one_df.values).flatten()

    # 创建散点图
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, alpha=0.6)

    # 设置坐标轴范围
    plt.xlim(0, 500)  # 设置x轴范围为0-500
    plt.ylim(0, 6)    # 设置y轴范围为0-6

    # 添加标题和轴标签
    plt.title('Scatter Plot of Absolute Values')
    plt.xlabel('Index')
    plt.ylabel('Absolute Value')

    # 添加网格线以提高可读性
    plt.grid(True, linestyle='--', alpha=0.7)
    # 保存为PNG格式的图片
    plt.savefig('output/scatter_plot_seed%d.png'%(args.seed), dpi=300, bbox_inches='tight')

def save_csv(n_by_one_df):
    n_by_one_df.to_csv("output/output.csv", index=False, header=False)
    # print("csv文件已保存")
    # print(n_by_one_df)

# class optuna_param_demo:
#     import optuna
#     def objective(trial):
#         x = trial.suggest_uniform('x', -10, 10)
#         return (x - 2) ** 2


#     if __name__ == '__main__':
#         study = optuna.create_study()
#         study.optimize(objective, n_trials=100)
#         print('Best value: {} (params: {})\n'.format(study.best_value, study.best_params))


for seed in range(15,20):
    args.seed = 20
    print(args)
    import random
    import numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    #%%
    print("Loading dataset seed is %d"%args.seed)
    print(args.name)
    from Dataset import load_smd_smap_msl, loader_SWat, loader_WADI, loader_PSM, loader_WADI_OCC

    if args.name == 'SWaT':
        train_loader, val_loader, test_loader, n_sensor = loader_SWat(args.data_dir, \
                                                                        args.batch_size, args.window_size, args.stride_size, args.train_split)

    elif args.name == 'Wadi':
        train_loader, val_loader, test_loader, n_sensor = loader_WADI(args.data_dir, \
                                                                    args.batch_size, args.window_size, args.stride_size, args.train_split)

    elif args.name == 'SMAP' or args.name == 'MSL' or args.name.startswith('machine'):
        train_loader, val_loader, test_loader, n_sensor = load_smd_smap_msl(args.name, \
                                                                    args.batch_size, args.window_size, args.stride_size, args.train_split)

    elif args.name == 'PSM':
        train_loader, val_loader, test_loader, n_sensor = loader_PSM(args.name, \
                                                                    args.batch_size, args.window_size, args.stride_size, args.train_split)



    #%%
    model = MTGFLOW(args.n_blocks, args.input_size, args.hidden_size, args.n_hidden, args.window_size, n_sensor, dropout=0.0, model = args.model, batch_norm=args.batch_norm)
    model = model.to(device)

    #%%
    from torch.nn.utils import clip_grad_value_
    import seaborn as sns
    import matplotlib
    matplotlib.use('Agg')  # 或者 'TkAgg'
    import matplotlib.pyplot as plt
    save_path = os.path.join(args.output_dir,args.name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)


    loss_best = 100
    roc_max = 0
  
    lr = args.lr 
    '''
    optimizer用一个就可，用多个会出问题
    '''
    optimizer = torch.optim.Adam([
        {'params':model.parameters(), 'weight_decay':args.weight_decay},
        ], lr=lr, weight_decay=0.0)

    for epoch in range(40):
        # print(epoch)
        loss_train = []

        model.train()
        for x,_,idx in train_loader:
            x = x.to(device)

            optimizer.zero_grad()
            '''
            model根据时间序列的数量，例化多个
            '''
            loss = -model(x,) # 返回负的似然估计平均值,此处再加个负号变为正数,符合我们的逻辑
            '''
            不同loss之间直接相加
            '''
            total_loss = loss

            total_loss.backward()
            clip_grad_value_(model.parameters(), 1)
            optimizer.step()
            loss_train.append(loss.item())



        loss_test = []
        with torch.no_grad():
            for x,_,idx in test_loader:

                x = x.to(device)
                loss = -model.test(x, ).cpu().numpy()
                loss_test.append(loss)
        loss_test = np.concatenate(loss_test)

        # print("aaaaaaa")
        # print(loss_test)
        # unique_labels, counts = np.unique(np.asarray(test_loader.dataset.label, dtype=int), return_counts=True)
        # print("Unique labels:", unique_labels)
        # print("Label counts:", counts)
    
        # roc_test = roc_auc_score(np.asarray(test_loader.dataset.label,dtype=int),loss_test)

    
        # if roc_max < roc_test:
        #     roc_max = roc_test
        #     torch.save({
        #     'model': model.state_dict(),
        #     }, f"{save_path}/model.pth")

        # roc_max = max(roc_test, roc_max)
        if epoch == 39:
            print("====================ROUND%d====================="%(epoch))
            # print(roc_max)
            # print(loss_test)
            # 转dataframe
            n_by_one_df = pd.DataFrame([loss_test])
            save_png(n_by_one_df)
            save_csv(n_by_one_df)
            print("================================================\n\n")

