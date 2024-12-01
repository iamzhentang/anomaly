#%%
import os
import argparse
import torch
from models.MTGFLOW import MTGFLOW
import numpy as np
# from sklearn.metrics import roc_auc_score, precision_recall_curve 
# import pandas as pd
from torch.nn.utils import clip_grad_value_
from Dataset import load_smd_smap_msl, loader_SWat, loader_WADI, loader_PSM, loader_WADI_OCC


import optuna
from optuna.trial import TrialState
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import random
import numpy as np



def objective(trial):
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
    # parser.add_argument('--batch_norm', type=bool, default=True)
    parser.add_argument('--train_split', type=float, default=0.6)
    parser.add_argument('--stride_size', type=int, default=10)

    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--window_size', type=int, default=60)
    parser.add_argument('--lr', type=float, default=2e-3, help='Learning rate.')



    args = parser.parse_known_args()[0]
    args.cuda = torch.cuda.is_available()
    DEVICE = torch.device("cuda" if args.cuda else "cpu")
    BATCHSIZE = args.batch_size
    EPOCHS = 10
    N_TRAIN_EXAMPLES = BATCHSIZE * 30
    N_VALID_EXAMPLES = BATCHSIZE * 10


    seed=15
    args.seed = seed
    # print(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    #%%
    print("Loading dataset seed is %d"%seed)
    print(args.name)
    # Generate optuna sampliing
    n_blocks_t = trial.suggest_int("n_blocks", 1, 10)
    hidden_size_t = trial.suggest_int("hidden_size", 16, 128)
    n_hidden_t = trial.suggest_int("n_hidden", 1, 3)
    # batch size \ window size
    # input_size_t = trial.suggest_categorical("input_size", [1, 2, 3])
    # dropout_t = trial.suggest_float("dropout",0, 1e-1)
    optimizer_name_t = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr_t = trial.suggest_float("lr", 1e-5, 1e-1, log=True)

    # get data
    train_loader, valid_loader, test_loader, n_sensor = loader_WADI(args.data_dir, \
                                                            args.batch_size, args.window_size, args.stride_size, args.train_split)
    

    # Generate the model.
    # model = MTGFLOW(args.n_blocks, args.input_size, args.hidden_size, args.n_hidden, args.window_size, n_sensor, dropout=0.0, model = args.model, batch_norm=args.batch_norm)
    model = MTGFLOW(n_blocks_t, args.input_size, hidden_size_t, n_hidden_t, args.window_size, n_sensor, dropout=0.0, model=args.model, batch_norm=True)

    model = model.to(DEVICE)

    # Generate the optimizers.
    optimizer = getattr(optim, optimizer_name_t)(model.parameters(), lr=lr_t)

    # Training of the model.
    for epoch in range(EPOCHS):

        model.train()
        train_loss = 0
        n_train_batches = 0
        for batch_idx, (x,_,idx) in enumerate(train_loader):
            # Limiting training data for faster epochs.
            if batch_idx * BATCHSIZE >= N_TRAIN_EXAMPLES:
                break
            x = x.to(DEVICE)

            optimizer.zero_grad()
            loss = -model(x,) # 返回负的似然估计平均值,此处再加个负号变为正数,符合我们的逻辑
            print("train_loss is",loss)
            total_loss = loss
            total_loss.backward()
            clip_grad_value_(model.parameters(), 1)
            # clip loss or normalize loss
            optimizer.step()
            # loss_train.append(loss.item())
            train_loss += loss.item()
            n_train_batches += 1


        model.eval()
        valid_loss = 0
        n_valid_batches = 0
        with torch.no_grad():
            for batch_idx, (x,_,idx) in enumerate(valid_loader):
                # Limiting validation data.
                if batch_idx * BATCHSIZE >= N_VALID_EXAMPLES:
                    break
                x = x.to(DEVICE)
                loss = -model.test(x, ).cpu().numpy()
                print("val_loss is",loss)
                if isinstance(loss,(list,np.ndarray)):
                    loss = loss[0]
                    # print("loss is list")
                else:
                    loss = loss
                    # print("loss is not list")
                valid_loss += loss.item()
                n_valid_batches += 1

        # 计算平均验证loss
        avg_valid_loss = valid_loss / n_valid_batches if n_valid_batches > 0 else float('inf')
        
        # 报告验证loss而不是准确率
        trial.report(avg_valid_loss, epoch)
        
        # 基于验证loss进行剪枝（注意：因为是loss，所以是越小越好）
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    # 返回最终的验证loss作为优化目标
    return avg_valid_loss



if __name__ == "__main__":
    # storage_name = "sqlite:///optuna.db"
    # study = optuna.create_study(
    #     pruner=optuna.pruners.MedianPruner(n_warmup_steps=3), direction="minimize",
    #     study_name="MTGFLOW", storage=storage_name,load_if_exists=True
    # )
    study = optuna.create_study(
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=3), direction="maximize",
        study_name="MTGFLOW",load_if_exists=True
    )
    
    study.optimize(objective, n_trials=20, timeout=1200)

    best_params = study.best_params
    best_value = study.best_value
    print("\n\nbest_value = "+str(best_value))
    print("best_params:")
    print(best_params)