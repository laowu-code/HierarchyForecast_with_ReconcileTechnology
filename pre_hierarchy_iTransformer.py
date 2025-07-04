import argparse
import csv
import time
import pandas as pd
import dill
import numpy as np
import torch
from matplotlib import pyplot as plt
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
# from models.iTransformer import iTransformer_single, iTransformer_block
# from models import KAN
from models import hierarchy_iTransformer_model
# import models
from data import split_data, data_detime
from utils.tools import metrics_of_pv, EarlyStopping, same_seeds, train, evaluate
from utils import ql_loss, save_dict_to_csv
import os
import warnings
import json
import optuna
from utils import reconcile_all, get_average_metrics

warnings.filterwarnings('ignore')

if __name__ == "__main__":
    for site, dataset in zip(['Tibet', 'Tibet', 'USA', 'Australia'], ['2018', '2019', '2019', '2019']):
        # for site, dataset in zip(['Australia', ], ['2019']):
        seeds = 42
        same_seeds(seeds)
        site = site
        dataset = dataset
        # site = 'USA'
        # site = 'Australia'

        # dataset = '2018'

        parser = argparse.ArgumentParser(description="Hyperparameters")
        parser.add_argument("--batch_size", type=int, default=300)
        parser.add_argument("--learning_rate", type=float, default=0.001)
        parser.add_argument("--epochs", type=int, default=50)
        # parser.add_argument('--data_dir', type=str, default='./dataset', help='数据集的路径')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        args = parser.parse_args()
        batch_size = args.batch_size
        learning_rate = args.learning_rate
        epochs = args.epochs
        file_path = f'./data/{site}/{site}_{dataset}.csv'
        num_nodes = 5
        epoch = 200
        time_length = 24 * 2
        # predict_length = [1,4]
        predict_length = 6
        device = torch.device('cuda:0')
        df_all = pd.read_csv(file_path, header=0).iloc[2:21746, 5].values
        multi_steps = True

        data_train, data_valid, data_test, scalar = split_data(df_all, 17396, 2174)
        dataset_train = data_detime(data=data_train, lookback_length=time_length, multi_steps=multi_steps,
                                    lookforward_length=predict_length)
        dataset_valid = data_detime(data=data_valid, lookback_length=time_length, multi_steps=multi_steps,
                                    lookforward_length=predict_length)
        dataset_test = data_detime(data=data_test, lookback_length=time_length, multi_steps=multi_steps,
                                   lookforward_length=predict_length)

        train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(dataset_valid, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)


        # model = Model_Torder(input_size=5).to(device)
        # model = Model_Iorder(input_size=5).to(device)
        # model = TCN(input_size=5, output_size=1, num_channels=[32] * 3, kernel_size=3).to(device)

        def objective(trial):

            # dim_embed = trial.suggest_categorical('hidden_dim', [16, 32, 64, 128, 256, 512])
            # layer_ = trial.suggest_categorical('layer_I', [1, 2, 3, 4, 5, 6])
            # heads = trial.suggest_categorical('heads', [2, 4, 6, 8, 12])
            dict_parameters = {
                'Tibet_2018': [32, 1, 6],
                'Tibet_2019': [32, 2, 6],
                'USA_2019': [64, 1, 6],
                'Australia_2019': [64, 1, 4],
            }
            dim_embed = 64
            layer_ = 1
            heads = 6
            # need_train = True
            need_train = False
            # reconcile_flag = False
            reconcile_flag = True
            if not need_train:
                model = hierarchy_iTransformer_model(time_length, predict_length,
                                                     dim=dict_parameters[f'{site}_{dataset}'][0],
                                                     depth=dict_parameters[f'{site}_{dataset}'][1],
                                                     heads=dict_parameters[f'{site}_{dataset}'][2]).to(device)
            else:
                model = hierarchy_iTransformer_model(time_length, predict_length, dim=dim_embed, depth=layer_,
                                                     heads=heads).to(device)
            criterion_QL = ql_loss
            optm = optim.Adam(model.parameters(), lr=learning_rate)
            optm_schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optm, mode="min", factor=0.5, patience=5, verbose=True)
            model_name = f"hierarchy_iTransformer_{site}_{dataset}"
            model_save = f"model_save/{site}/{dataset}/{model_name}.pt"
            train_losses, valid_losses = [], []
            earlystopping = EarlyStopping(model_save, patience=10, delta=0.0001)
            if not os.path.exists(f"model_save/{site}/{dataset}"):
                os.makedirs(f"model_save/{site}/{dataset}")
                os.makedirs(f"model_save/{site}/{dataset}/best")

            model_save = f"model_save/{site}/{dataset}/{model_name}.pt" if need_train else f"model_save/{site}/{dataset}/best/{model_name}.pt"
            if need_train:
                try:
                    for epoch in range(epochs):
                        time_start = time.time()
                        train_loss = train(data=train_loader, model=model, criterion=criterion_QL, optm=optm, )
                        valid_loss, ms = evaluate(data=valid_loader, model=model, criterion=criterion_QL, )
                        train_losses.append(train_loss)
                        valid_losses.append(valid_loss)
                        optm_schedule.step(valid_loss)
                        earlystopping(valid_loss, model)
                        # torch.save(model, model_save, pickle_module=dill)
                        print('')
                        print(
                            f'Epoch:{epoch + 1}| {model_name}|time:{(time.time() - time_start):.2f}|Loss_train:{train_loss:.4f}|Learning_rate:{optm.state_dict()["param_groups"][0]["lr"]:.4f}\n'
                            f'Loss_valid:{valid_loss:.4f}|{ms}',
                            flush=True, )
                        if earlystopping.early_stop:
                            print("Early stopping")
                            break  # 跳出迭代，结束训练
                except KeyboardInterrupt:
                    print("Training interrupted by user")
                # plt.plot(np.arange(len(train_losses)), train_losses, label="train loss")
                # plt.plot(np.arange(len(valid_losses)), valid_losses, label="valid rmse")----------------*
                # plt.legend()  # 显示图例
                # plt.xlabel("epoches")
                # # plt.ylabel("epoch")
                # plt.title("Train_loss&Valid_loss")
                # plt.show()
            with open(model_save, "rb") as f:
                # model = torch.load(f, pickle_module=dill)
                model.load_state_dict(torch.load(f))
            # print(model)
            model = model.to(device)
            test_loss, ms_test = evaluate(
                data=test_loader, model=model, criterion=criterion_QL, scalar=scalar, )
            ms_test['model_name'] = model_name
            print(
                f'Test_valid:{test_loss:.4f}| base forecasts :{ms_test}', )
            # with open(f'data_record/{site}/{dataset}/Metrics_{model_name}.json', 'a', newline='') as f:
            #     json.dump(ms_test, f, indent=4)
            save_dict_to_csv(ms_test,
                             f'data_record/{site}/{dataset}/Base_Metrics_{model_name}.xlsx')
            if reconcile_flag:
                S = [[1, 0, 0, 0, 0, 0],
                     [0, 1, 0, 0, 0, 0],
                     [0, 0, 1, 0, 0, 0],
                     [0, 0, 0, 1, 0, 0],
                     [0, 0, 0, 0, 1, 0],
                     [0, 0, 0, 0, 0, 1],
                     [1 / 3, 1 / 3, 1 / 3, 0, 0, 0],
                     [0, 0, 0, 1 / 3, 1 / 3, 1 / 3],
                     [1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6]]
                S = np.array(S)
                _, _, y_pred_train, y_true_train = evaluate(data=train_loader, model=model, criterion=criterion_QL,
                                                            scalar=scalar, flag=reconcile_flag)
                _, _, y_pred_test, y_true_test = evaluate(data=test_loader, model=model, criterion=criterion_QL,
                                                          scalar=scalar, flag=reconcile_flag)
                data_save_dict = {}
                data_save_dict['base'] = y_pred_test[:, :, 0]
                data_save_dict['true'] = y_true_test
                G = reconcile_all(y_true_train[:, :], y_pred_train[:, :, 0], S=S)
                reconcile_method = ['BU', 'TD', 'ols', 'hls_add', 'wls', 'mint_sample', 'mint_shrink']
                for i, g in enumerate(G):
                    y_pred_reconcile = S @ g @ y_pred_test
                    data_save_dict[reconcile_method[i]] = y_pred_reconcile[:, :, 0]
                    metrics_dict_tmp = get_average_metrics(y_pred_reconcile, y_true_test)
                    metrics_dict_tmp['model_name'] = reconcile_method[i] + '_' + model_name
                    save_dict_to_csv(metrics_dict_tmp,
                                     f'data_record/{site}/{dataset}/Reconcile_Metrics_{model_name}.xlsx')
                np.savez(f'data_record/{site}/{dataset}/Forecasts_{model_name}.npz', **data_save_dict)

            return None if not need_train else valid_loss


        study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(), load_if_exists=True,
                                    storage=f'sqlite:///data_record/db_wind.sqlite3',
                                    study_name=f'hierarchy_iTransformer_{site}_{dataset}')
        study.optimize(objective, n_trials=1)
        print(study.best_params, '\n', study.best_value)
    # csv_write = csv.writer(f)
    # csv_write.writerow([f'{site}_pred1_{model_name}', ms_test[0], ms_test[1], ms_test[2], ms_test[3]])
# optuna-dashboard sqlite:///data_record/db_wind.sqlite3
