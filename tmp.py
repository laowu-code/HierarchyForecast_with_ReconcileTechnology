import numpy as np
import torch
import math


def cp( label, up, low):  # PICP‘s algorithm is the same
    l, n = label.shape
    picp = np.zeros((n))
    for i in range(n):
        result1 = (label[:, i] <= up[:, i]).astype(int)
        result2 = (label[:, i] >= low[:, i]).astype(int)
        result = result1 + result2
        picp[i] = np.round(np.sum(result == 2, axis=0) / l, 4)
    return picp


def pinaw( label, up, low):
    l,n=label.shape
    pinaw = np.zeros((n))
    for i in range(n):
        pinaw[i] = np.round(np.mean(np.abs(up[:,i] - low[:,i]), axis=0) / (np.max(label[:,i]) - np.min(label[:,i])),4)
    return pinaw

def skill_score_single(label, up, low, alpha):
    SC1 = 0
    SC2 = 0
    SC3 = 0
    SC = []
    cc = up - low
    for i in range(len(label)):
        if label[i] > up[i]:
            a = (-2 * alpha * cc[i] - 4 * (label[i] - up[i]))
            SC1 += a
            SC.append(a)
        elif label[i] >= low[i] and label[i] <= up[i]:
            a = -2 * alpha * cc[i]
            SC2 += a
            SC.append(a)
        else:
            a = -2 * alpha * cc[i] - 4 * (low[i] - label[i])
            SC3 += a
            SC.append(a)
    return np.round((SC1 + SC2 + SC3) / len(label), 4), np.round(SC, 4)

a=np.array([[0.95,0.96,0.98],[0.95,0.96,0.98]])
b=np.array([[1,1.5,1.7],[1.1,1.6,1.8]])
c=np.array([[1.5,2,2.4],[1.5,2,2.4]])
# print(pinaw(b,c,a,))

import numpy as np

def pinball_loss(label, up, low, quantile_low, quantile_high):
    """
    Calculate pinball loss for multi-dimensional input.

    Args:
        label (np.ndarray): True values, shape (N, L).
        up (np.ndarray): Upper bounds, shape (N, L).
        low (np.ndarray): Lower bounds, shape (N, L).
        quantile_low (float): Quantile level for lower bounds.
        quantile_high (float): Quantile level for upper bounds.

    Returns:
        tuple:
            - Average pinball loss for each column (L-dimensional array).
            - Pinball loss for all elements (N x L array).
    """
    N, L = label.shape

    # Calculate the loss components for lower bounds
    loss_low = (label - low) * (quantile_low * (low <= label) - (1 - quantile_low) * (low > label))

    # Calculate the loss components for upper bounds
    loss_up = (label - up) * (quantile_high * (up <= label) - (1 - quantile_high) * (up > label))

    # Total loss for each element
    loss = (loss_low + loss_up) / 2

    # Average pinball loss for each column
    avg_loss = np.round(np.mean(loss, axis=0), 4)

    # Return results
    return avg_loss, np.round(loss, 4)
def skill_score(label, up, low, alpha):
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
def pinball_loss_single( label, up, low, quantile_low, quantile_high):

    PL = (np.sum(
            (label - low) * (quantile_low * (low <= label) - (1 - quantile_low) * (low > label)), axis=0) + np.sum(
            (label - up) * (quantile_high * (up <= label) - (1 - quantile_high) * (up > label)), axis=0)) / (2 * label.shape[0])

    return np.round(PL, 4)


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
# Example usage
if __name__ == "__main__":
    # Example input (N=5, L=3)
    # label = np.array([[1.0, 2.0, 3.0], [1.5, 2.5, 3.5], [2.0, 3.0, 4.0], [0.5, 1.5, 2.5], [1.0, 2.0, 3.0]])
    # up = np.array([[2.0, 3.0, 4.0], [2.5, 3.5, 4.5], [3.0, 4.0, 5.0], [1.5, 2.5, 3.5], [2.0, 3.0, 4.0]])
    # low = np.array([[0.9, 1.5, 2.5], [1.0, 2.0, 3.0], [1.8, 2.5, 3.5], [0.0, 1.0, 2.0], [0.5, 1.5, 2.5]])
    # alpha = 0.1
    #
    # # Calculate skill scores
    # a,b=pinball_loss(label, up, low,0.1,0.9)
    # print(pinball_loss_single(label, up, low,0.1,0.9))
    # print(a,'\n',b)
    # avg_score, all_scores = skill_score(label, up, low, alpha)
    # is1,is2=skill_score_single(label[:,0],up[:,0],low[:,0],alpha)
    # print(is1,is2)
    # print("Average Skill Score (per column):", avg_score)
    # print("All Skill Scores (N x L):")
    # print(all_scores)
    # x=torch.Tensor([[i+1 for i in range(6)] for j in range(2)])
    # a=np.random.randn(2,6)
    # b=np.random.randn(2,3)
    # k1,k2=np.concatenate([a, b], axis=1).shape
    # print(k1,k2)
    # print(a[:,-1,None].shape)
    # print(np.expand_dims(np.mean(a,axis=1),axis=1).shape)
    # # print(y_expand(x))
    # y_dict={f'index{i}':[i]*3 for i in range(5)}
    # print(y_dict)
    # for site, dataset in zip(['Tibet', 'Tibet', 'USA', 'Australia'], ['2018', '2019', '2019', '2019']):
    #     print(site, dataset)
    import pandas as pd
    import os


    # def save_dict_to_csv(data_dict, file_name):
    #     # Extract the 'node_average' and 'level_average' sections
    #     node_average = data_dict.get('node_average', {})
    #     level_average = data_dict.get('level_average', {})
    #     model_name = data_dict.get('Model Name', 'Unknown Model')
    #
    #     # Create point DataFrame
    #     def create_point_df(data):
    #         point_data = data.get('point', {})
    #         df = pd.DataFrame([point_data], index=[model_name])
    #         return df
    #
    #     # Create probability DataFrame
    #     def create_prob_df(data):
    #         prob_data = data.get('prob', {})
    #         columns = []
    #         for conf in ['confidence_95', 'confidence_90', 'confidence_85']:
    #             for metric in ['PICP', 'ACE', 'PINAW', 'CWC', 'IS', 'PL']:
    #                 columns.append(f"{metric}_{conf[-2:]}")
    #         prob_values = []
    #         for conf in ['confidence_95', 'confidence_90', 'confidence_85']:
    #             prob_values.extend(prob_data.get(conf, []))
    #         df = pd.DataFrame([prob_values], columns=columns, index=[model_name])
    #         return df
    #
    #     # Process node_average
    #     node_point_df = create_point_df(node_average)
    #     node_prob_df = create_prob_df(node_average)
    #     node_combined_df = pd.concat([node_point_df, node_prob_df], axis=1)
    #
    #     # Process level_average
    #     level_point_df = create_point_df(level_average)
    #     level_prob_df = create_prob_df(level_average)
    #     level_combined_df = pd.concat([level_point_df, level_prob_df], axis=1)
    #
    #     # Save to an Excel file with two sheets
    #     if os.path.exists(file_name):
    #         with pd.ExcelWriter(file_name, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
    #             node_combined_df.to_excel(writer, sheet_name='node_average',
    #                                       startrow=writer.sheets['node_average'].max_row, header=False)
    #             level_combined_df.to_excel(writer, sheet_name='level_average',
    #                                        startrow=writer.sheets['level_average'].max_row, header=False)
    #     else:
    #         with pd.ExcelWriter(file_name, engine='xlsxwriter') as writer:
    #             node_combined_df.to_excel(writer, sheet_name='node_average')
    #             level_combined_df.to_excel(writer, sheet_name='level_average')


    # Example usage
    # data = {
    #     'node_average': {
    #         'point': {'MAE': 0.1682, 'R2': 0.9434, 'RMSE': 0.2883, 'MMAPE': 0.0515},
    #         'prob': {
    #             'confidence_95': [0.8985, -0.0515, 0.117, 3.0267, -0.2027, 0.0253],
    #             'confidence_90': [0.8496, -0.0504, 0.0912, 3.1599, -0.2865, 0.0358],
    #             'confidence_85': [0.7996, -0.0504, 0.0737, 3.6794, -0.348, 0.0435],
    #         },
    #     },
    #     'level_average': {
    #         'point': {'MAE': 0.1844, 'R2': 0.9364, 'RMSE': 0.3085, 'MMAPE': 0.0567},
    #         'prob': {
    #             'confidence_95': [0.8925, -0.0575, 0.1313, 3.72, -0.2162, 0.027],
    #             'confidence_90': [0.8345, -0.0655, 0.0996, 5.6461, -0.3135, 0.0392],
    #             'confidence_85': [0.782, -0.068, 0.0799, 7.1244, -0.3817, 0.0477],
    #         },
    #     },
    #     'Model Name': 'hierarchy_PatchTST_Australia_2019',
    # }

    # save_dict_to_csv(data, 'output_file.xlsx')
    x=np.random.rand(1000, 9)
    x2=np.random.rand(1000, 9)
    a="tmp"
    # np.savez(f'data_record/tmp.npz', a=x)
    dict_tmp={}
    s=['s1','s2','s3','s4','s5','s6']
    for i,x in enumerate(s):
        dict_tmp[x]=i*2
    np.savez(f'data_record/tmp.npz', **dict_tmp)
    x=np.load(f'data_record/tmp.npz',)
    x1=x['s4']
    if x2.all()==x1.all():
        print("Success")



