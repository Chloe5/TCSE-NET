import torch.nn as nn
import torch
import numpy as np
import math
import pickle
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import time
import dataload
import torch.utils.data as Data

#*********data**********
observation = 3 * 3600 -1
n_time_interval = 6
time_interval = math.ceil((observation+1)*1.0/n_time_interval)
batch_size = 2

hidden_dim = (32,)
kernel_size = (2,)
num_layers = 1
input_dim = 100
dense1 = 32
dense2 = 16

data_path ="D:\\学术相关\\007.CasCN-master\\dataset_weibo"

import ChebyLSTM

batch_first = True
model = ChebyLSTM.MODEL(input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first, n_time_interval, dense1, dense2)

criterion = nn.MSELoss(reduction= 'mean')

opt1 = torch.optim.Adam(model.parameters(),lr=1e-3)


start = time.time()
step = 0
train_step = 0
train_loss = []
val_step = 0
val_loss = []
test_step = 0
test_loss = []
display_step =1600
max_try = 10
patience = max_try
lr = 0.0005

for i in range(130):
    if train_step / 160 ==0:
        input = math.floor(train_step/160)
        filepath = data_path + '\\data_train\\data_train_' + str(input) + '.pkl'
        data_train = dataload.MyDataset(filepath, n_time_interval, time_interval)
        train_step = 0

        data_train = Data.dataloader(data_train, batch_size =batch_size)

    for id, batch in enumerate(data_train):
        batch_x, batch_L, n_steps, batch_y, batch_time_interval, batch_rnn_index = batch

    train_step += 1
    opt1.zero_grad()

    pred = model(batch_x, batch_L, n_steps,
                hidden_dim, batch_time_interval, batch_rnn_index)

    loss = criterion(pred, batch_y)
    print('train_loss', loss)
    train_loss.append(loss)

    loss.backward()
    opt1.step()

    if step / display_step == 0:
        with torch.no_grad:
            if val_step / 160 == 0:
                input = math.floor(val_step / 160)
                filepath = data_path + '\\val_train\\val_train_' + str(input) + '.pkl'
                data_val = dataload.MyDataset(filepath, n_time_interval, time_interval)
                val_step =0
                data_val = Data.dataloader(data_train, batch_size=batch_size)

            for id, batch in enumerate(data_val):
                val_x, val_L, n_steps, val_y, val_time_interval, val_rnn_index = batch

            val_step += 1

            model.eval()
            val_pred = model(val_x, val_L, n_steps,
                             hidden_dim, val_time_interval, val_rnn_index)

            Val_loss = criterion(val_pred, val_y)

            val_loss.append(val_loss)

            if np.mean(val_loss) < best_val_loss:
                best_val_loss = np.mean(val_loss)
                best_test_loss = np.mean(test_loss)
                patience = max_try

            predict_result = []
            if test_step / 160 == 0:
                input = math.floor(test_step / 160)
                filepath = data_path + '\\test_train\\test_train_' + str(input) + '.pkl'
                data_test = dataload.MyDataset(filepath, n_time_interval, time_interval)
                test_step = 0
                data_test = Data.dataloader(data_train, batch_size=batch_size)

            for id, batch in enumerate(data_test):
                test_x, test_L, n_steps, test_y, test_time_interval, test_rnn_index = batch

            test_step += 1
            test_loss = []
            model.eval()
            test_pred = model(val_x, val_L, n_steps,
                             hidden_dim, val_time_interval, val_rnn_index)

            test_loss = criterion(val_pred, val_y)

            test_loss.append(test_loss)

            print("last test error:", np.mean(test_loss))
            pickle.dump((predict_result, test_y, test_loss), open(
                "prediction_result_" + str(lr) + "_CasCN", 'wb'))
            print("#" + str(step / display_step) +
                  ", Training Loss= " + "{:.6f}".format(np.mean(train_loss)) +
                  ", Validation Loss= " + "{:.6f}".format(np.mean(val_loss)) +
                  ", Test Loss= " + "{:.6f}".format(np.mean(test_loss)) +
                  ", Best Valid Loss= " + "{:.6f}".format(best_val_loss) +
                  ", Best Test Loss= " + "{:.6f}".format(best_test_loss)
                  )
            train_loss = []
            patience -= 1
            if not patience:
                break
        step += 1

print(len(predict_result), len(test_y))
print("Finished!\n----------------------------------------------------------------")
print("Time:", time.time() - start)
print("Valid Loss:", best_val_loss)
print("Test Loss:", best_test_loss)
