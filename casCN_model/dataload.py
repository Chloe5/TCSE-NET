import torch.utils.data as Data
import torch
import pickle
import numpy as np
import math
import gc

#filepath = 'D:\\学术相关\\007.CasCN-master\\dataset_weibo'
class MyDataset(Data.Dataset):
    def __init__(self, filepath, n_time_interval, time_interval):
        # 获得训练数据的总行
        _, x, L, y, sz, time, _ = pickle.load(open(filepath, 'rb'))

        self.number = len(x)
        batch_x = []
        batch_L = []
        batch_y =  np.zeros(shape=(len(x), 1))

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

            batch_time_interval_index_sample = []
            num_step = len(x[i])
            self.n_steps = num_step

            for j in range(sz[i]):
                temp_time = np.zeros(shape=(n_time_interval))
                k = int(math.floor(time[i][j] / time_interval))
                temp_time[k] = 1
                batch_time_interval_index_sample.append(temp_time)
            if len(batch_time_interval_index_sample) < num_step:
                for n in range(num_step - len(batch_time_interval_index_sample)):
                    temp_time_padding = np.zeros(shape=(n_time_interval))
                    batch_time_interval_index_sample.append(temp_time_padding)
                    n = n + 1
            batch_time_interval_index.append(batch_time_interval_index_sample)
            rnn_index_temp = np.zeros(shape=(num_step))
            rnn_index_temp[:sz[i]] = 1
            batch_rnn_index.append(rnn_index_temp)

        self.x = torch.tensor(batch_x)
        del(batch_x)
        self.L = torch.tensor(batch_L)
        del(batch_L)
        self.y = torch.tensor(batch_y)
        del(batch_y)
        self.time_interval_index = torch.tensor(batch_time_interval_index)
        del(batch_time_interval_index)
        self.rnn_index = torch.tensor(batch_rnn_index)
        del(batch_rnn_index)
        gc.collect()

    def __len__(self):
        return self.number
    def __getitem__(self, idx):
        return self.x[idx], self.L[idx],self.n_steps, self.y[idx], self.time_interval_index[idx], self.rnn_index[idx]
