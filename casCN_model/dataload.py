import torch.utils.data as Data
import torch
import pickle
import numpy as np
import math
import gc

#filepath = 'D:\\学术相关\\007.CasCN-master\\dataset_weibo'
class MyDataset(Data.Dataset):
    def __init__(self, filepath, n_time_interval):
        # 获得训练数据的总行
        _, x, L, y, sz, time, _ = pickle.load(open(filepath, 'rb'))

        self.number = len(x)
        batch_x = []
        batch_L = []
        batch_y =  np.zeros(shape=(len(x), 1))
        rnn_index = []

        batch_time_interval_index = []
        batch_rnn_index = []
        for i in range(len(x)):
            # x
            temp_x = []
            for k in range(len(x[i])):
                temp_x.append(x[i][k].todense().tolist())
            batch_x.append(temp_x)
            batch_L.append(L[i].todense().tolist())
            batch_y[i,0] = y[i]

            rnn_index.append(time[i].reshape(-1))


        self.x = torch.tensor(batch_x)
        self.L = torch.tensor(batch_L)
        self.y = torch.tensor(batch_y)
        batch_time_interval_index_sample = torch.eye(n_time_interval)
        self.time_interval_index = batch_time_interval_index_sample.expand((len(x),n_time_interval,n_time_interval))
        self.rnn_index = torch.tensor(rnn_index)

        gc.collect()

    def __len__(self):
        return self.number

    def __getitem__(self, idx):
        return self.x[idx], self.L[idx], self.y[idx], self.time_interval_index[idx], self.rnn_index[idx]
