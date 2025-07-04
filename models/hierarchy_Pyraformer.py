import torch.nn as nn
import torch
from models.Pyraformer import Pyraformer


class hierarchy_Pyraformer_model(nn.Module):
    def __init__(self, time_length, pred_length=6,d_ff=256,d_model=128,n_heads=6,e_layers=3,):
        super(hierarchy_Pyraformer_model, self).__init__()
        self.base_model1 = Pyraformer(pred_len=pred_length*7,seq_len=time_length,d_ff=d_ff,n_heads=n_heads,
                                      e_layers=e_layers,d_model=d_model,enc_in=1,)  # 原始输入的模型

        self.base_model2 = Pyraformer(pred_len=int(pred_length/3)*7,seq_len=int(time_length/3),d_ff=d_ff,n_heads=n_heads,
                                      e_layers=e_layers,d_model=d_model,enc_in=1,)  # 池化大小为3的模型

        self.base_model3 = Pyraformer(pred_len=int(pred_length/6)*7,seq_len=int(time_length/6),d_ff=d_ff,n_heads=n_heads,
                                      e_layers=e_layers,d_model=d_model,enc_in=1,)  # 池化大小为6的模型

        self.pool3 = nn.AvgPool1d(kernel_size=3, stride=3)  # 平均池化，大小为3
        self.pool6 = nn.AvgPool1d(kernel_size=6, stride=6)  # 平均池化，大小为6

    def forward(self, x):
        # 确保输入是二维或三维 (batch_size, sequence_length,n_var)
        # 子模型1：直接输入
        out1 = self.base_model1(x).reshape(x.shape[0], -1, 7)

        # 子模型2：池化大小为3的输入
        pooled_x3 = self.pool3(x.transpose(2, 1)).transpose(2, 1)
        out2 = self.base_model2(pooled_x3).reshape(x.shape[0], -1, 7)

        # 子模型3：池化大小为6的输入
        pooled_x6 = self.pool6(x.transpose(2, 1)).transpose(2, 1)
        out3 = self.base_model3(pooled_x6).reshape(x.shape[0], -1, 7)
        out = torch.cat([out1, out2, out3], dim=1)

        return out
