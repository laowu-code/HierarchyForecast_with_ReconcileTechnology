import os

import pandas as pd
import torch
from matplotlib import pyplot as plt
import dill
from tqdm import tqdm
import numpy as np
import math
import properscoring as ps
import scipy.stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt


def metrics_of_pv(preds, trues):
    pred = np.array(preds)
    true = np.array(trues)
    mae = np.round(mean_absolute_error(true, pred), 4)
    rmse = np.round(np.sqrt(mean_squared_error(true, pred)), 4)
    r2 = np.round(r2_score(true, pred), 4)
    mbe = np.round(np.mean(pred - true), 4)
    # sMAPE = np.round(100 * np.mean(np.abs(preds - trues) / (np.abs(preds) + np.abs(trues))), 4)
    return [mae, rmse, r2, mbe]





def save2csv(n, file):
    n = n.reshape((1, n.shape[0]))
    n = pd.DataFrame(n)
    n.to_csv(file, index=False, encoding='utf-8', header=False, mode='a')


def same_seeds(seed):
    torch.manual_seed(seed)  # 固定随机种子（CPU）
    if torch.cuda.is_available():  # 固定随机种子（GPU)
        torch.cuda.manual_seed(seed)  # 为当前GPU设置
        torch.cuda.manual_seed_all(seed)  # 为所有GPU设置
    np.random.seed(seed)  # 保证后续使用random函数时，产生固定的随机数
    torch.backends.cudnn.benchmark = True  # GPU、网络结构固定，可设置为True
    # torch.backends.cudnn.deterministic = True  # 固定网络结构


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, save_path, patience=7, verbose=False, delta=0):
        """
        Args:
            save_path : 模型保存文件夹
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.save_path = save_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        # path = os.path.join(self.save_path, 'best_network.pth')
        path = self.save_path
        torch.save(model.state_dict(), path, pickle_module=dill)  # 这里会存储迄今最优模型的参数
        self.val_loss_min = val_loss

def y_expand(y):
    """
    :param y: N*6
    :return: N*9
    """
    y_mean1=torch.mean(y[:,:3],dim=1)
    y_mean2=torch.mean(y[:,3:6],dim=1)
    y_mean3=torch.mean(y[:,:],dim=1)
    y_expand=torch.cat([y,y_mean1.unsqueeze(1),y_mean2.unsqueeze(1),y_mean3.unsqueeze(1)],dim=1)
    return y_expand

def train(data, model, criterion, optm, device=torch.device("cuda:0")):
    model.train()
    running_loss = 0.0
    for x, y in tqdm(data):
        model.zero_grad()
        x, y = x.float().to(device), y.float().to(device)
        optm.zero_grad()
        y=y_expand(y)
        y_pre = model(x)
        loss = criterion(y_pre, y)
        loss.backward()
        optm.step()
        running_loss += loss.item() * x.size(0)
    epoch_loss = running_loss / len(data.dataset)
    return epoch_loss


def evaluate(data, model, criterion, device=torch.device("cuda:0"), scalar=None,flag=False):
    model.eval()
    val_running_loss = 0.0
    all_preds = None
    all_labels = None
    for x, y in tqdm(data):
        model.zero_grad()
        with torch.no_grad():
            x, y = x.float().to(device), y.float().to(device)
            y = y_expand(y)
            y_pre = model(x)
            loss = criterion(y_pre, y)
            val_running_loss += loss.item() * x.size(0)
            if all_preds is None:
                all_preds = y_pre.cpu().numpy()
                all_labels = y.cpu().numpy()
            else:
                all_preds = np.concatenate((all_preds, y_pre.cpu().numpy()), axis=0)
                all_labels = np.concatenate((all_labels, y.cpu().numpy()), axis=0)
            # all_preds.extend(y_pre.cpu().numpy())
            # all_labels.extend(y.cpu().numpy())
    epoch_loss = val_running_loss / len(data.dataset)
    if scalar is not None:
        _,p,l=all_preds.shape
        all_preds = scalar.inverse_transform(all_preds.reshape(all_preds.shape[0],-1)).reshape(-1,p,l)
        all_labels = scalar.inverse_transform(all_labels)
        # l = 300
        # x = np.arange(len(all_preds[0:l]))
        # plt.plot(x, all_preds[0:l], color='#FB873B', linewidth=2, label='Forecasts')
        # # 绘制第二条线（对应数组b），设置颜色为蓝色（'b'），线条粗细为2（可根据实际需求调整）
        # plt.plot(x, all_labels[0:l], color='k', linestyle='--', linewidth=2, label='Observed')
        # # 添加标题
        # plt.title('PV power forecasting')
        # # 添加坐标轴标签
        # plt.xlabel('Time(h)')
        # plt.ylabel('Value(KW)')
        # # 添加图例，使图标显示出来，loc参数指定图例位置，这里设置为最佳位置自动调整
        # plt.legend(loc='upper right')
        # # 展示图形
        #
        # plt.savefig('./pic_/pv_power_forecast.svg', format='SVG', dpi=600, transparent=True)
        # plt.show()
    metrics_dict = get_average_metrics(all_preds, all_labels)
    if flag:
        return epoch_loss, metrics_dict,all_preds,all_labels
    else:
        return epoch_loss, metrics_dict


# import CRPS.CRPS as pscore
def metrics_windspeed(y_pred, y_true):
    """
    Calculate MAE, R2, RMSE, and MMAPE for each prediction step.

    Args:
        y_pred (np.ndarray): Predicted values, shape (N, P, 7).
            The first column is the point prediction.
        y_true (np.ndarray): True values, shape (N, P).

    Returns:
        dict: A dictionary containing metrics for each step:
            - 'MAE': Mean Absolute Error, shape (P,).
            - 'R2': R-squared, shape (P,).
            - 'RMSE': Root Mean Squared Error, shape (P,).
            - 'MMAPE': Mean Maximum Absolute Percentage Error, shape (P,).
    """
    # Extract the point predictions from y_pred (first column)
    point_preds = y_pred[:, :, 0]  # Shape (N, P)

    # Number of prediction steps
    N, P = y_true.shape

    # Initialize metrics
    mae = np.zeros(P)
    r2 = np.zeros(P)
    rmse = np.zeros(P)
    mmape = np.zeros(P)

    # Loop through each step
    for p in range(P):
        y_pred_p = point_preds[:, p]  # Predicted values for step p
        y_true_p = y_true[:, p]  # True values for step p

        # MAE
        mae[p] = np.mean(np.abs(y_pred_p - y_true_p))

        # R2
        ss_total = np.sum((y_true_p - np.mean(y_true_p)) ** 2)
        ss_residual = np.sum((y_true_p - y_pred_p) ** 2)
        r2[p] = 1 - ss_residual / ss_total if ss_total != 0 else 0

        # RMSE
        rmse[p] = np.sqrt(np.mean((y_pred_p - y_true_p) ** 2))

        # MMAPE
        mmape[p] = np.mean(np.abs(y_pred_p - y_true_p) / np.maximum(np.abs(y_true_p), 1e-8))

    # Compile results into a dictionary
    metrics = {
        'MAE': np.round(mae, 4),
        'R2': np.round(r2, 4),
        'RMSE': np.round(rmse, 4),
        'MMAPE': np.round(mmape, 4),
    }

    return metrics

class prob_metric(object):

    def get_metrics(self, quantile_low, quantile_high, lower, upper, label, isList=False):
        # quantile_low下边界对应的分位数；quantile_high为上边界对应的置信度；lower为下边界；upper为上边界；label:标签
        # label_data = label.flatten()
        # lower = lower.flatten()
        # upper = upper.flatten()
        label_data = label
        lower = lower
        upper = upper
        u = quantile_high - quantile_low
        PL = self.pinball_loss(label_data, upper, lower, quantile_low, quantile_high)
        # WQL = self.wQL(label_data, upper, lower, quantile_low, quantile_high)
        PICP = self.cp(label_data, upper, lower)  # PICP越接近置信度越好（0，1）
        # MWP = self.mwp(label_data, upper, lower)
        # MC = self.mc(MWP, CP)
        PINAW = self.pinaw(label_data, upper, lower)
        # PINRW = self.pinrw(label_data, upper, lower)
        CWC = self.cwc(PICP, PINAW, 50, u)
        IS, IS_list = self.skill_score(label_data, upper, lower, 1 - u)
        ACE = np.round((PICP - u), 4)

        return PICP, ACE, PINAW, CWC, IS, PL

    def pinball_loss(self, label, up, low, quantile_low, quantile_high):

        PL = (np.sum(
            (label - low) * (quantile_low * (low <= label) - (1 - quantile_low) * (low > label)), axis=0) + np.sum(
            (label - up) * (quantile_high * (up <= label) - (1 - quantile_high) * (up > label)), axis=0)) / (
                     2 * label.shape[0])

        return np.round(PL, 4)

    def wQL(self, label, up, low, quantile_low, quantile_high):
        PL = (np.sum(
            (label - low) * (quantile_low * (low <= label) - (1 - quantile_low) * (low > label)), axis=0) + np.sum(
            (label - up) * (quantile_high * (up <= label) - (1 - quantile_high) * (up > label)), axis=0))
        wql = 2 * PL / np.sum(label)
        return np.round(wql, 4)

    def cp(self, label, up, low):  # PICP‘s algorithm is the same
        l, n = label.shape
        picp = np.zeros((n))
        for i in range(n):
            result1 = (label[:, i] <= up[:, i]).astype(int)
            result2 = (label[:, i] >= low[:, i]).astype(int)
            result = result1 + result2
            picp[i] = np.round(np.sum(result == 2, axis=0) / l, 4)
        return picp

    def mwp(self, label, up, low):
        mwp = np.mean(np.abs(up - low) / label)
        return round(mwp, 4)

    def mc(self, mwp, cp):
        mc = mwp / cp
        return np.round(mc, 4)

    def pinaw(self, label, up, low):
        l, n = label.shape
        pinaw = np.zeros((n))
        for i in range(n):
            pinaw[i] = np.round(
                np.mean(np.abs(up[:, i] - low[:, i]), axis=0) / (np.max(label[:, i]) - np.min(label[:, i])), 4)
        return pinaw

    def pinrw(self, label, up, low):
        PINRW = np.sqrt(np.mean(np.square(up - low))) / (np.max(label) - np.min(label))
        return np.round(PINRW, 4)

    # def cwc(self, index_picp, index_pinaw, n, u):
    #     e = math.exp(-n * (index_picp - [u]*12))
    #     if index_picp >= u:
    #         r = 0
    #     else:
    #         r = 1
    #     index = index_pinaw + r * e
    #     return np.round(index, 4)

    def cwc(self, index_picp, index_pinaw, n, u):
        l = (-n * (index_picp - u))
        e = np.zeros(l.shape[0])
        r = np.zeros(l.shape[0])
        for i in range(l.shape[0]):
            e[i] = math.exp(l[i])
            if index_picp[i] >= u:
                r[i] = 0
            else:
                r[i] = 1
        index = index_pinaw * (1 + r * e)
        return np.round(index, 4)

    def skill_score(self, label, up, low, alpha):
        """
        Calculate skill score for inputs with shape (N, L).

        Args:
            label (np.ndarray): True values, shape (N, L).
            up (np.ndarray): Upper bounds, shape (N, L).
            low (np.ndarray): Lower bounds, shape (N, L).
            alpha (float): Weight parameter.

        Returns:
            tuple:
                - Average skill score for each column (L-dimensional array).
                - Skill scores for all elements (N, L array).
        """
        # Calculate coverage (cc)
        cc = up - low
        # Case 1: label > up
        case1_mask = label > up
        sc1 = np.where(case1_mask, -2 * alpha * cc - 4 * (label - up), 0)
        # Case 2: low <= label <= up
        case2_mask = (label >= low) & (label <= up)
        sc2 = np.where(case2_mask, -2 * alpha * cc, 0)
        # Case 3: label < low
        case3_mask = label < low
        sc3 = np.where(case3_mask, -2 * alpha * cc - 4 * (low - label), 0)

        # Combine all cases
        sc = sc1 + sc2 + sc3
        # Average skill score for each column
        avg_skill_score = np.round(np.mean(sc, axis=0), 4)
        # Return results
        return avg_skill_score, np.round(sc, 4)

    # pred_result:[样本个数，每个样本的各个分位点的条件分位数]；quantiles：一个array向量，存储的是各个分为点的值
    def ProbabilityPredictionMetricCalculation(self, pred_result, quantiles, label):
        label_data = label.flatten()
        crps = self.CRPS(pred_result, label_data, quantiles)
        print("CRPS:{}".format(crps))
        return crps

    def cdf(self, pred_result, quantiles):
        y_cdf = np.zeros((pred_result.shape[0], quantiles.size + 2))
        y_cdf[:, 1:-1] = pred_result
        y_cdf[:, 0] = 2.0 * pred_result[:, 1] - pred_result[:, 2]
        y_cdf[:, -1] = 2.0 * pred_result[:, -2] - pred_result[:, -3]
        qs = np.zeros((1, quantiles.size + 2))
        qs[0, 1:-1] = quantiles
        qs[0, 0] = 0.0
        qs[0, -1] = 1.0
        return y_cdf, qs

    def CRPS(self, pred_result, label, quantiles):
        y_cdf, qs = self.cdf(pred_result, quantiles)
        ind = np.zeros(y_cdf.shape)
        ind[y_cdf > label.reshape(-1, 1)] = 1.0
        CRPS = np.trapz((qs - ind) ** 2.0, y_cdf)
        CRPS = np.mean(CRPS)
        return round(CRPS, 4)

    def CRPS_LPLS(selfself, pred_result, label):

        return None


def probabilistic_metrics(y_pred, y_true):
    """
    y_pred: N*l*n_tau
    y_true: N*l
    """
    metrics = prob_metric()
    PICP_95, ACE_95, PINAW_95, CWC_95, IS_95, PL_95 = metrics.get_metrics(0.025, 0.975, y_pred[:, :, 1],
                                                                          y_pred[:, :, -1], y_true)
    PICP_90, ACE_90, PINAW_90, CWC_90, IS_90, PL_90 = metrics.get_metrics(0.05, 0.95, y_pred[:, :, 2],
                                                                          y_pred[:, :, -2], y_true)
    PICP_85, ACE_85, PINAW_85, CWC_85, IS_85, PL_85 = metrics.get_metrics(0.075, 0.925, y_pred[:, :, 3],
                                                                          y_pred[:, :, -3], y_true)
    return {'confidence_95': np.array([PICP_95, ACE_95, PINAW_95, CWC_95, IS_95, PL_95]),
            'confidence_90': np.array([PICP_90, ACE_90, PINAW_90, CWC_90, IS_90, PL_90]),
            'confidence_85': np.array([PICP_85, ACE_85, PINAW_85, CWC_85, IS_85, PL_85])}

def convert_numpy_to_list(obj):
    if isinstance(obj, np.ndarray):
        # 使用 np.around 函数对数组元素保留四位小数，再转换为列表
        return np.around(obj, decimals=4).tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_list(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_to_list(item) for item in obj]
    return obj

def get_average_metrics(y_pred, y_true):
    point_dict = metrics_windspeed(y_pred, y_true)
    prob_dict = probabilistic_metrics(y_pred, y_true)
    metrics_dict={}
    print(point_dict['MAE'].shape,'\n',prob_dict['confidence_95'].shape)
    point_ = {key: np.round(np.mean(value),4) for key, value in point_dict.items()}
    prob_ = {key: np.mean(value, axis=1) for key, value in prob_dict.items()}
    metrics_dict['node_average']={'point':point_,'prob':prob_}

    point_ = {key: np.round(np.mean([np.mean(value[:6]), np.mean(value[6:8]), value[-1]]),4) for key, value in
                  point_dict.items()}
    prob_ = {key: np.mean(np.concatenate([np.expand_dims(np.mean(value[:, :6], axis=1),axis=1), np.expand_dims(np.mean(value[:,6:8], axis=1),axis=1), value[:,-1,None]],axis=1), axis=1) for
                 key, value in prob_dict.items()}
    metrics_dict['level_average'] = {'point': point_, 'prob': prob_}
    return convert_numpy_to_list(metrics_dict)



# def IP_Indicator(y_pre_quantile, y_real):
#     Error = np.ndarray(shape=(3, 6))
#     Metric_all = prob_metric()
#     quantile_low, quantile_high = 0.075, 0.925
#     lower, upper, label = y_pre_quantile[:, 2].reshape(len(y_real), 1), \
#         y_pre_quantile[:, 3].reshape(len(y_real), 1), \
#         y_real.reshape(len(y_real), 1)
#     Error[0, 0], Error[0, 1], Error[0, 2], Error[0, 3], Error[0, 4], Error[0, 5] = \
#         Metric_all.get_metrics(quantile_low, quantile_high, lower, upper, label)
#     lower85, upper85 = lower, upper
#
#     quantile_low, quantile_high = 0.05, 0.95
#     lower, upper, label = y_pre_quantile[:, 1].reshape(len(y_real), 1), \
#         y_pre_quantile[:, 4].reshape(len(y_real), 1), \
#         y_real.reshape(len(y_real), 1)
#     Error[1, 0], Error[1, 1], Error[1, 2], Error[1, 3], Error[1, 4], Error[1, 5] = \
#         Metric_all.get_metrics(quantile_low, quantile_high, lower, upper, label)
#     lower90, upper90 = lower, upper
#
#     quantile_low, quantile_high = 0.025, 0.975
#     lower, upper, label = y_pre_quantile[:, 0].reshape(len(y_real), 1), \
#         y_pre_quantile[:, 5].reshape(len(y_real), 1), \
#         y_real.reshape(len(y_real), 1)
#     Error[2, 0], Error[2, 1], Error[2, 2], Error[2, 3], Error[2, 4], Error[2, 5] = \
#         Metric_all.IntervalPredictionMetricCalculation(quantile_low, quantile_high, lower, upper, label)
#     lower95, upper95 = lower, upper
#     All_quantile = np.vstack([lower85.T, upper85.T, lower90.T, upper90.T, lower95.T, upper95.T]).T
#     return Error, All_quantile


# def calculate_index_probability(y_predict, y_ture, confidence, scaler=None):
#     metric_all = metric()
#     quantile_low, quantile_high = round((1 - confidence) / 2, 4), round((1 + confidence) / 2, 4)
#     # y_predict[:, :, 0]=scaler.inverse_transform(y_predict[:, :, 0])
#     # y_predict[:, :, 1] = scaler.scale_*y_predict[:, :, 1]
#     (lower, upper) = scipy.stats.laplace.interval(confidence, loc=y_predict[:, :, 0],
#                                                   scale=y_predict[:, :, 1] / sqrt(y_predict.shape[0]))
#     if scaler is not None:
#         lower = scaler.inverse_transform(lower)
#         upper = scaler.inverse_transform(upper)
#     index_all = metric_all.IntervalPredictionMetricCalculation(quantile_low=quantile_low, quantile_high=quantile_high,
#                                                                lower=lower, upper=upper, label=y_ture)
#     return index_all

def point_error(inv_y_va, inv_yhat_va):
    error = np.zeros([4, ])
    mse_va = mean_squared_error(inv_y_va, inv_yhat_va)
    rmse_va = sqrt(mean_squared_error(inv_y_va, inv_yhat_va))
    mae_va = mean_absolute_error(inv_y_va, inv_yhat_va)
    r_square_va = r2_score(inv_y_va, inv_yhat_va)
    error[0], error[1], error[2], error[3] = mse_va, rmse_va, mae_va, r_square_va
    print('r_square: %.6f' % r_square_va)
    print('mse: %.6f' % mse_va)
    print('rmse: %.6f' % rmse_va)
    print('mae: %.6f' % mae_va)
    return error


def CRPS_Gauss(mu, sigma, tures):
    mu = mu
    sigma = sigma
    num_samples, num_steps = mu.shape[0], mu.shape[1]
    crps_12a = np.zeros(num_steps)
    # for i in range(num_steps):
    #     for j in range(num_samples):
    #         pass
    for i in range(num_steps):
        crps_12a[i] = ps.crps_gaussian(tures[:, i], mu=mu[:, i], sig=sigma[:, i]).mean()
        # for j in range(num_samples):
        #     tmp.append(ps.crps_quadrature(tures[j, i], scipy.stats.norm(loc=mu[j, i], scale=sigma[j, i])))
        # crps_12b[i] = np.array(tmp).mean()
        # tmp = []
    return np.round(crps_12a, 4), np.round(crps_12a.mean(), 4)


def CRPS_Student(df, loc, scale, tures):
    df = df
    loc = loc
    scale = scale
    tmp = []
    num_samples, num_steps = df.shape[0], df.shape[1]
    crps_12 = np.zeros(num_steps)
    for i in range(num_steps):
        for j in range(num_samples):
            # tmp.append(ps.crps_quadrature(tures[j, i], scipy.stats.laplace(loc=mu[j, i], scale=np.sqrt(2 * sigma[j,i] ** 2))))
            # ensemble=scipy.stats.laplace.rvs(loc=mu[j, i], scale=np.sqrt(2 * sigma[j,i] ** 2),size=1000)
            ensemble = scipy.stats.t.rvs(df=df[j, i], loc=loc[j, i], scale=scale[j, i], size=1000)
            # ensemble = scipy.stats.norm.rvs(loc=mu[j, i], scale= sigma[j, i], size=1000)
            tmp.append(ps.crps_ensemble(tures[j, i], ensemble))
        crps_12[i] = np.array(tmp).mean()
        tmp = []
    return np.round(crps_12, 4), np.round(crps_12.mean(), 4)


def CRPS_LPLS(mu, sigma, tures):
    mu = mu
    sigma = sigma
    tmp = []
    num_samples, num_steps = mu.shape[0], mu.shape[1]
    crps_12 = np.zeros(num_steps)
    for i in range(num_steps):
        for j in range(num_samples):
            # tmp.append(ps.crps_quadrature(tures[j, i], scipy.stats.laplace(loc=mu[j, i], scale=np.sqrt(2 * sigma[j,i] ** 2))))
            ensemble = scipy.stats.laplace.rvs(loc=mu[j, i], scale=np.sqrt(2 * sigma[j, i] ** 2), size=1000)
            # ensemble = scipy.stats.norm.rvs(loc=mu[j, i], scale= sigma[j, i], size=1000)
            tmp.append(ps.crps_ensemble(tures[j, i], ensemble))
        crps_12[i] = np.array(tmp).mean()
        tmp = []
    return np.round(crps_12, 4), np.round(crps_12.mean(), 4)


def save_dict_to_csv(data_dict, file_name):
    # Extract the 'node_average' and 'level_average' sections
    node_average = data_dict.get('node_average', {})
    level_average = data_dict.get('level_average', {})
    model_name = data_dict.get('model_name', 'Unknown Model')

    # Create point DataFrame
    def create_point_df(data):
        point_data = data.get('point', {})
        df = pd.DataFrame([point_data], index=[model_name])
        return df

    # Create probability DataFrame
    def create_prob_df(data):
        prob_data = data.get('prob', {})
        columns = []
        for conf in ['confidence_95', 'confidence_90', 'confidence_85']:
            for metric in ['PICP', 'ACE', 'PINAW', 'CWC', 'IS', 'PL']:
                columns.append(f"{metric}_{conf[-2:]}")
        prob_values = []
        for conf in ['confidence_95', 'confidence_90', 'confidence_85']:
            prob_values.extend(prob_data.get(conf, []))
        df = pd.DataFrame([prob_values], columns=columns, index=[model_name])
        return df

    # Process node_average
    node_point_df = create_point_df(node_average)
    node_prob_df = create_prob_df(node_average)
    node_combined_df = pd.concat([node_point_df, node_prob_df], axis=1)

    # Process level_average
    level_point_df = create_point_df(level_average)
    level_prob_df = create_prob_df(level_average)
    level_combined_df = pd.concat([level_point_df, level_prob_df], axis=1)

    # Save to an Excel file with two sheets
    if os.path.exists(file_name):
        with pd.ExcelWriter(file_name, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
            node_combined_df.to_excel(writer, sheet_name='node_average',
                                      startrow=writer.sheets['node_average'].max_row, header=False)
            level_combined_df.to_excel(writer, sheet_name='level_average',
                                       startrow=writer.sheets['level_average'].max_row, header=False)
    else:
        with pd.ExcelWriter(file_name, engine='xlsxwriter') as writer:
            node_combined_df.to_excel(writer, sheet_name='node_average')
            level_combined_df.to_excel(writer, sheet_name='level_average')
