
#%%
from cgitb import reset
from turtle import forward, shape
from numpy import percentile
import torch.nn as nn
import torch.nn.functional as F
from models.NF import MAF
import torch

def interpolate(tensor, index, target_size, mode = 'nearest', dim = 0):
    print(tensor.shape)
    source_length = tensor.shape[dim]
    if source_length > target_size:
        raise AttributeError('no need to interpolate')
    if dim == -1:
        new_tensor = torch.zeros((*tensor.shape[:-1], target_size),dtype=tensor.dtype, device=tensor.device)
    if dim == 0:
        new_tensor = torch.zeros((target_size, *tensor.shape[1:], ),dtype=tensor.dtype, device=tensor.device)
    scale = target_size // source_length
    reset = target_size % source_length
    # if mode == 'nearest':
    new_index = index
    new_tensor[new_index, :] = tensor
    new_tensor[:new_index[0], :] = tensor[0,:].unsqueeze(0)
    for i in range(source_length-1):
        new_tensor[new_index[i]:new_index[i+1] , :] = tensor[i,:].unsqueeze(0)
    new_tensor[new_index[i+1] :,:] = tensor[i+1,:].unsqueeze(0)
    return new_tensor

class GNN(nn.Module):
    """
    The GNN module applied in GANF
    """
    def __init__(self, input_size, hidden_size):

        super(GNN, self).__init__()
        self.lin_n = nn.Linear(input_size, hidden_size)
        self.lin_r = nn.Linear(input_size, hidden_size, bias=False)
        self.lin_2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, h, A):
        ## A: K · K
        ## H: N · K  · L · D
        # print(h.shape, A.shape)
        # h_n = self.lin_n(torch.einsum('nkld,kj->njld',h,A))
        # h_n = self.lin_n(torch.einsum('nkld,kj->njld',h,A))
        # print(h.shape, A.shape)
        h_n = self.lin_n(torch.einsum('nkld,nkj->njld',h,A))
        '''
        获取窗口下所有t时刻及其历史信息的时空条件：
        h_r：只考虑序列中除了最后一个时间步的所有时间步——>实体本身的历史信息有助于增强时间序列的时间关系
        h_n：将递归特征添加到 h_n 的相应位置，从第二个时间步开始
        实现了t时刻和t-1时刻的历史信息相加，实现公式11
        '''
        h_r = self.lin_r(h[:,:,:-1])
        h_n[:,:,1:] += h_r
        h = self.lin_2(F.relu(h_n))

        return h

import math
import torch.nn as nn
import matplotlib.pyplot as plt
def plot_attention(data, i, X_label=None, Y_label=None):
  '''
    Plot the attention model heatmap
    Args:
      data: attn_matrix with shape [ty, tx], cutted before 'PAD'
      X_label: list of size tx, encoder tags
      Y_label: list of size ty, decoder tags
  '''
  fig, ax = plt.subplots(figsize=(20, 8)) # set figure size
  heatmap = ax.pcolor(data, cmap=plt.cm.Blues, alpha=0.9)
  fig.colorbar(heatmap)
  # Set axis labels
  if X_label != None and Y_label != None:
    X_label = [x_label for x_label in X_label]
    Y_label = [y_label for y_label in Y_label]
    
    xticks = range(0,len(X_label))
    ax.set_xticks(xticks, minor=False) # major ticks
    ax.set_xticklabels(X_label, minor = False, rotation=45)   # labels should be 'unicode'
    
    yticks = range(0,len(Y_label))
    ax.set_yticks(yticks, minor=False)
    ax.set_yticklabels(Y_label[::-1], minor = False)   # labels should be 'unicode'
    
    ax.grid(True)
    plt.show()
    plt.savefig('graph/attention{:04d}.jpg'.format(i))



class ScaleDotProductAttention(nn.Module):
    """
    compute scale dot product attention

    Query : given sentence that we focused on (decoder)
    Key : every sentence to check relationship with Qeury(encoder)
    Value : every sentence same with Key (encoder)
    """

    def __init__(self, c):
        super(ScaleDotProductAttention, self).__init__()
        self.w_q = nn.Linear(c, c)
        self.w_k = nn.Linear(c, c)
        self.w_v = nn.Linear(c, c)
        self.softmax = nn.Softmax(dim = 1)
        self.dropout = nn.Dropout(0.2)
        # swat_0.2
    def forward(self, x,mask=None, e=1e-12):
        # input is 4 dimension tensor
        # [batch_size, head, length, d_tensor]
        shape = x.shape
        x_shape = x.reshape((shape[0],shape[1], -1))
        batch_size, length, c = x_shape.size()
        q = self.w_q(x_shape)
        k = self.w_k(x_shape)
        k_t = k.view(batch_size, c, length)  # transpose
        score = (q @ k_t) / math.sqrt(c)  # scaled dot product

        # 2. apply masking (opt)
        if mask is not None:
            score = score.masked_fill(mask == 0, -1e9)

        # 3. pass them softmax to make [0, 1] range
        score = self.dropout(self.softmax(score))



        return score, k


class MTGFLOW(nn.Module):

    def __init__ (self, n_blocks, input_size, hidden_size, n_hidden, window_size, n_sensor, dropout = 0.1, model="MAF", batch_norm=True):
        super(MTGFLOW, self).__init__()

        self.rnn = nn.LSTM(input_size=input_size,hidden_size=hidden_size,batch_first=True, dropout=dropout)
        self.gcn = GNN(input_size=hidden_size, hidden_size=hidden_size)
        if model=="MAF":
            # self.nf = MAF(n_blocks, n_sensor, input_size, hidden_size, n_hidden, cond_label_size=hidden_size, batch_norm=batch_norm,activation='tanh', mode = 'zero')
            self.nf = MAF(n_blocks, n_sensor, input_size, hidden_size, n_hidden, cond_label_size=hidden_size, batch_norm=batch_norm,activation='tanh')
      
        self.attention = ScaleDotProductAttention(window_size*input_size)
    def forward(self, x, ):

        return self.test(x, ).mean()

    def test(self, x, ):
        # x: N · K · L · D 
        # N 表示batch中的样本数量
        # 𝐾 表示每个样本的实体总数
        # 𝐿 表示每个实体的观测总数
        # D 表示每个观测的特征数量
        # x shape [batch_size, head, length, d_tensor]
        full_shape = x.shape # 有4个维度，N K L D
        # print(full_shape)
        graph,_ = self.attention(x) #返回score，key，学习邻居矩阵A
        self.graph = graph
        # reshape: N*K, L, D
        x = x.reshape((x.shape[0]*x.shape[1], x.shape[2], x.shape[3]))
        h,_ = self.rnn(x) # 只保留整个序列的信息 h，h包含了输入序列x的时间动态信息
        '''
        h 是 LSTM 层的输出，它是一个三维张量，其形状为 (num_layers * num_directions, batch_size, hidden_size)。
        num_layers 是 LSTM 网络的层数。
        num_directions 表示 LSTM 是否是双向的，单向为 1，双向为 2。
        hidden_size 是 LSTM 层的隐藏状态大小，这个大小通常由模型设计者指定，用于捕捉输入数据的高级特征表示。
        '''

        # resahpe: N, K, L, H
        h = h.reshape((full_shape[0], full_shape[1], h.shape[1], h.shape[2]))# 把num_layers * num_directions丢掉，不需要
        h = self.gcn(h, graph)# x的特征经过LSTM提取后，用h替换掉x的特征。新的x与邻居矩阵A卷积——>实现公式11

        # reshappe N*K*L,H
        h = h.reshape((-1,h.shape[3]))# 展平为二维数据，其中第一个维度是批次中的元素总数，第二个维度是每个元素的LSTM后的特征数。
        # reshappe N*K*L,D
        x = x.reshape((-1,full_shape[3]))# 展平为二维数据，其中第一个维度是批次中的元素总数，第二个维度是每个元素LSTM前的特征数。
        '''
        #该方法计算了输入 x 和 h 的对数概率
        然后使用MAF进行归一化流
        full_shape[1]：是实体（窗口）的总数
        full_shape[2]：是log_prob的window_size
        h：是LSTM提取的特征，是y
        vvvvvvvvvvvvvvvvvvvvv'''
        # TODO
        # log_prob = self.nf.log_prob(x, full_shape[1], full_shape[2], h).reshape([full_shape[0],-1])
        log_prob = self.nf.log_prob(x, full_shape[1], full_shape[2]).reshape([full_shape[0],-1])
        # log_prob重塑为一个二维张量，其中第一个维度是批次大小，第二个维度是自动计算的。以便可以进一步处理或用于损失计算
        log_prob = log_prob.mean(dim=1)

        return log_prob

    def get_graph(self):
        return self.graph

    def locate(self, x, ):
        # x: N X K X L X D 
        full_shape = x.shape

        graph, _ = self.attention(x)
        # reshape: N*K, L, D
        self.graph = graph
        x = x.reshape((x.shape[0]*x.shape[1], x.shape[2], x.shape[3]))
        h,_ = self.rnn(x)

        # resahpe: N, K, L, H
        h = h.reshape((full_shape[0], full_shape[1], h.shape[1], h.shape[2]))
        h = self.gcn(h, graph)

        # reshappe N*K*L,H
        h = h.reshape((-1,h.shape[3]))
        x = x.reshape((-1,full_shape[3]))
        a = self.nf.log_prob(x, full_shape[1], full_shape[2], h)
        log_prob, z = a[0].reshape([full_shape[0],full_shape[1],-1]), a[1].reshape([full_shape[0],full_shape[1],-1])
        


        return log_prob.mean(dim=2), z.reshape((full_shape[0]* full_shape[1],-1))


class test(nn.Module):
    def __init__ (self, n_blocks, input_size, hidden_size, n_hidden, window_size, n_sensor, dropout = 0.1, model="MAF", batch_norm=True):
        super(test, self).__init__()
        
        if model=="MAF":
            self.nf = MAF(n_blocks, n_sensor, input_size, hidden_size, n_hidden, batch_norm=batch_norm,activation='tanh', mode='zero')
        self.attention = ScaleDotProductAttention(window_size*input_size)
    def forward(self, x, ):
        return self.test(x, ).mean()
    def test(self, x):
        x = x.unsqueeze(2).unsqueeze(3)
        full_shape = x.shape
        x = x.reshape((full_shape[0]*full_shape[1], full_shape[2], full_shape[3]))
        x = x.reshape((-1,full_shape[3]))
        log_prob = self.nf.log_prob(x, full_shape[1], full_shape[2]).reshape([full_shape[0],full_shape[1],-1])#*full_shape[1]*full_shape[2]
        log_prob = log_prob.mean(dim=1)
        return log_prob

    def locate(self, x, ):
        # x: N X K X L X D 
        x = x.unsqueeze(2).unsqueeze(3)
        full_shape = x.shape
  
        x = x.reshape((x.shape[0]*x.shape[1], x.shape[2], x.shape[3]))

        # reshappe N*K*L,H
        x = x.reshape((-1,full_shape[3]))
        a = self.nf.log_prob(x, full_shape[1], full_shape[2])#*full_shape[1]*full_shape[2]
        log_prob, z = a[0].reshape([full_shape[0],full_shape[1],-1]), a[1].reshape([full_shape[0],full_shape[1],-1])
        

        return log_prob.mean(dim=2), z.reshape((full_shape[0]* full_shape[1],-1))


