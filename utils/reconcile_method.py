import numpy as np

def compute_mean(matrix, ctype='speed'):
   """
   将（num,6）转化为（num，9 or 10）
   """
   if ctype == 'speed':
      mean_first_three = np.mean(matrix[:, :3], axis=1, keepdims=True)
      mean_last_three = np.mean(matrix[:, 3:], axis=1, keepdims=True)
      mean_all = np.mean(matrix, axis=1, keepdims=True)
      result = np.concatenate((matrix, mean_first_three, mean_last_three, mean_all), axis=1)
   if ctype == 'power':
      mean_first_two = np.mean(matrix[:, :2], axis=1, keepdims=True)
      mean_middle_two = np.mean(matrix[:, 2:4], axis=1, keepdims=True)
      mean_last_two = np.mean(matrix[:, 4:], axis=1, keepdims=True)
      mean_all = np.mean(matrix, axis=1, keepdims=True)
      result = np.concatenate((matrix, mean_first_two, mean_middle_two, mean_last_two, mean_all), axis=1)
   return result


def cov2corr(cov, return_std=False):
   std_ = np.sqrt(np.diag(cov))
   corr = cov / np.outer(std_, std_)
   if return_std:
      return corr, std_
   else:
      return corr

def crossprod(x):
   return x.T @ x

def computeG_cov(true, pred, S, method):
   """
   Document can be found in https://robjhyndman.com/papers/MinT.pdf
   Code can be found in https://github.com/Nixtla/hierarchicalforecast/blob/main/hierarchicalforecast/methods.py
   这篇论文是高斯框架下的调和，并且也用了mint（sample）：Probabilistic Forecast Reconciliation under the Gaussian Framework
   """
   residuals = true - pred
   n_hiers, n_bottom = S.shape
   if method == 'ols':
      W = np.eye(n_hiers)
   elif method == 'hls':
      oneMat = np.ones((6, 1))
      W = np.diag(np.dot(S, oneMat).flatten())
   elif method == 'hls_add':
      W = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 3, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 3, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 6]])
   elif method == 'wls' or method == 'mint_sample' or method == 'mint_shrink':
      W1 = np.cov(residuals.T)
      if method == 'wls':
         W = np.diag(np.diag(W1))
      elif method == 'mint_sample':
         W = W1
      elif method == 'mint_shrink':
         W1D = np.diag(np.diag(W1))
         """
         Schäfer and Strimmer 2005, scale invariant shrinkage
         A Shrinkage Approach to Large-Scale Covariance Matrix Estimation and Implications for Functional Genomics
         """
         corm, residual_std = cov2corr(W1, return_std=True)
         corm = np.nan_to_num(corm, nan=0.0)
         residual_std = residual_std.reshape(-1, 1)
         xs = np.divide(residuals.T, residual_std, out=np.zeros_like(residuals.T), where=residual_std != 0)
         xs = xs[~np.isnan(xs).any(axis=1), :]
         n = n_hiers
         v = (1 / (n * (n - 1))) * (crossprod(xs ** 2) - (1 / n) * (crossprod(xs) ** 2))
         np.fill_diagonal(v, 0)
         corapn = cov2corr(W1D)
         corapn = np.nan_to_num(corapn, nan=0.0)
         d = (corm - corapn) ** 2
         lmd = v.sum() / d.sum()
         lmd = max(min(lmd, 1), 0)
         W = lmd * W1D + (1 - lmd) * W1
   else:
      print('method should be selected in ols, hls, hls_add, wls, mint_sample, mint_shrink.')
   G = np.dot(np.dot(np.linalg.inv(np.dot(np.dot(S.T, np.linalg.inv(W)), S)), S.T), np.linalg.inv(W))
   return G


def computeG(true, pred, S, method, ctype='speed'):
   """
   Document can be found in https://robjhyndman.com/papers/MinT.pdf
   Code can be found in https://github.com/Nixtla/hierarchicalforecast/blob/main/hierarchicalforecast/methods.py
   """
   residuals = true - pred
   n_hiers, n_bottom = S.shape
   if method == 'ols':
      W = np.eye(n_hiers)
   elif method == 'hls':
      oneMat = np.ones((6, 1))
      W = np.diag(np.dot(S, oneMat).flatten())
   elif method == 'hls_add':
      if ctype == 'speed':
         W = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 1, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 1, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 1, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 1, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 1, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 3, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 3, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 6]])
      if ctype == 'power':
         W = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 2, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 2, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 2, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 6]])
   elif method == 'wls' or method == 'mint_sample' or method == 'mint_shrink':
      W1 = np.dot(residuals.T, residuals)
      #         print('W1={}'.format(W1))
      if method == 'wls':
         W = np.diag(np.diag(W1))
      elif method == 'mint_sample':
         W = W1
      elif method == 'mint_shrink':
         W1D = np.diag(np.diag(W1))
         """
         Schäfer and Strimmer 2005, scale invariant shrinkage
         A Shrinkage Approach to Large-Scale Covariance Matrix Estimation and Implications for Functional Genomics
         """
         corm, residual_std = cov2corr(W1, return_std=True)
         corm = np.nan_to_num(corm, nan=0.0)
         residual_std = residual_std.reshape(-1, 1)
         xs = np.divide(residuals.T, residual_std, out=np.zeros_like(residuals.T), where=residual_std != 0)
         xs = xs[~np.isnan(xs).any(axis=1), :]
         n = n_hiers
         v = (1 / (n * (n - 1))) * (crossprod(xs ** 2) - (1 / n) * (crossprod(xs) ** 2))
         np.fill_diagonal(v, 0)
         corapn = cov2corr(W1D)
         corapn = np.nan_to_num(corapn, nan=0.0)
         d = (corm - corapn) ** 2
         lmd = v.sum() / d.sum()
         lmd = max(min(lmd, 1), 0)
         W = lmd * W1D + (1 - lmd) * W1
   else:
      print('method should be selected in ols, hls, hls_add, wls, mint_sample, mint_shrink.')
   G = np.dot(np.dot(np.linalg.inv(np.dot(np.dot(S.T, np.linalg.inv(W)), S)), S.T), np.linalg.inv(W))
   return G


def TopDown(history, method='prop_avg', ctype='speed'):
   """
   history:(num_sample,9)
   """
   if method == 'prop_avg':
      p = []
      s = np.sum(history, axis=0)
      for i in range(len(s)):
         p.append(s[i] / s[-1])
      p = np.array(p).reshape(-1, 1)  # (9,1)
   if method == 'avg_prop':
      num, f = history.shape
      p0 = np.zeros((num, f))
      for i in range(num):
         for j in range(f):
            p0[i, j] = history[i, j] / history[i, -1]
      p = np.mean(p0, axis=0).reshape(-1, 1)
   if ctype == 'speed':
      G = np.hstack((np.zeros((6, 8)), p[3:, :]))
   if ctype == 'power':
      G = np.hstack((np.zeros((6, 9)), p[4:, :]))
   return G


def TDFP(forecast):
   """
   基于预测比例的自上而下方法
   """
   p1 = 3 * forecast[:, 0] / (forecast[:, 0] + forecast[:, 1] + forecast[:, 2]) * 2 * forecast[:, 6] / (
              forecast[:, 6] + forecast[:, 7])
   p2 = 3 * forecast[:, 1] / (forecast[:, 0] + forecast[:, 1] + forecast[:, 2]) * 2 * forecast[:, 6] / (
              forecast[:, 6] + forecast[:, 7])
   p3 = 3 * forecast[:, 2] / (forecast[:, 0] + forecast[:, 1] + forecast[:, 2]) * 2 * forecast[:, 6] / (
              forecast[:, 6] + forecast[:, 7])
   p4 = 3 * forecast[:, 3] / (forecast[:, 3] + forecast[:, 4] + forecast[:, 5]) * 2 * forecast[:, 7] / (
              forecast[:, 6] + forecast[:, 7])
   p5 = 3 * forecast[:, 4] / (forecast[:, 3] + forecast[:, 4] + forecast[:, 5]) * 2 * forecast[:, 7] / (
              forecast[:, 6] + forecast[:, 7])
   p6 = 3 * forecast[:, 5] / (forecast[:, 3] + forecast[:, 4] + forecast[:, 5]) * 2 * forecast[:, 7] / (
              forecast[:, 6] + forecast[:, 7])
   GY = np.array(
      [forecast[:, 8] * p1, forecast[:, 8] * p2, forecast[:, 8] * p3, forecast[:, 8] * p4, forecast[:, 8] * p5,
       forecast[:, 8] * p6])
   return GY


def BottomUp(ctype='speed'):
   if ctype == 'speed':
      G = np.hstack((np.eye(6), np.zeros((6, 3))))
   if ctype == 'power':
      G = np.hstack((np.eye(6), np.zeros((6, 4))))
   return G


def MiddleOut(history, method):
   """
   history:(num_sample,9)
   """
   if method == 'prop_avg':
      s = np.sum(history, axis=0)
      p = np.array([s[0] / s[6], s[1] / s[6], s[2] / s[6], s[3] / s[7], s[4] / s[7], s[5] / s[7]])
   if method == 'avg_prop':
      num, f = history.shape
      p0 = np.zeros((num, 6))
      for i in range(num):
         p0[i, :] = np.array(
            [history[i, 0] / history[i, 6], history[i, 1] / history[i, 6], history[i, 2] / history[i, 6],
             history[i, 0] / history[i, 7], history[i, 1] / history[i, 7], history[i, 2] / history[i, 7]])
      p = np.mean(p0, axis=0)
   prop = np.array([[p[0], 0],
                    [p[1], 0],
                    [p[2], 0],
                    [0, p[3]],
                    [0, p[4]],
                    [0, p[5]]])
   G = np.hstack((np.zeros((6, 6)), prop, np.zeros((6, 1))))
   return G

S=[[1,0,0,0,0,0],
   [0,1,0,0,0,0],
   [0,0,1,0,0,0],
   [0,0,0,1,0,0],
   [0,0,0,0,1,0],
   [0,0,0,0,0,1],
   [1/3,1/3,1/3,0,0,0],
   [0,0,0,1/3,1/3,1/3],
   [1/6,1/6,1/6,1/6,1/6,1/6]]
S=np.array(S)
def reconcile_all(y_true,y_pred,S=S):
   G_all=[]
   G_all.append(BottomUp())# BU
   G_all.append(TopDown(history=y_true))#TD
   G_tmp=[computeG(y_true,y_pred,S,x) for x in ['ols', 'hls_add', 'wls', 'mint_sample', 'mint_shrink']]
   G_all = G_all + G_tmp
   return G_all




