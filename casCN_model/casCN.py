import torch.nn as nn
import torch
import numpy as np
import math
import pickle
import time
from casCN_model import dataload, ChebyLSTM
import torch.utils.data as Data
import os

#*********data**********
observation = 3 * 3600 -1
n_time_interval = 6 #
n_steps = 180 # 时续数量
time_interval = math.ceil((observation+1)*1.0/n_time_interval)
batch_size = 8

hidden_dim = (32,)
kernel_size = (2,)
num_layers = 1
input_dim = 100
dense1 = 32
dense2 = 16

data_path ="D:\\DKs-workshop\\canCN_pytorch\\dataset\\180_timeinterval"

batch_first = True
model = ChebyLSTM.MODEL(input_dim, hidden_dim, kernel_size, num_layers,
                        batch_first, n_time_interval, dense1, dense2)

criterion = nn.MSELoss(reduction= 'mean')

opt1 = torch.optim.Adam(model.parameters(),lr=1e-3)


start = time.time()
step = 0
train_step = 0
train_loss = []
val_loss = []
test_loss = []
display_step =1600
max_try = 10
patience = max_try
lr = 0.005
best_val_loss =10000
best_test_loss =10000
Epoch = 5

for e in range(Epoch):
    filepath = data_path + '\\data_train\\'
    filelist = os.listdir(filepath)
    for file in filelist:
        data_train = dataload.MyDataset(os.path.join(filepath,file), n_time_interval)
        batch_data_train = Data.DataLoader(data_train, batch_size =batch_size, drop_last= True)

        for id, batch in enumerate(batch_data_train):
            batch_x, batch_L, batch_y, batch_time_interval, batch_rnn_index = batch

            train_step += 1
            opt1.zero_grad()


            pred = model(batch_x, batch_L, n_steps,
                         hidden_dim, batch_rnn_index, batch_time_interval)

            print(pred)
            print(batch_y)
            loss = criterion(pred.float(), batch_y.float())

            train_loss.append(loss.tolist())
            print('train_loss：', np.mean(train_loss))

            opt1.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(parameters=model.parameters(), max_norm=5, norm_type=2)
            opt1.step()

    #evaluation and test
    with torch.no_grad():
        filepath = data_path + '\\data_val\\'
        filelist = os.listdir(filepath)
        for file in filelist:
            data_val = dataload.MyDataset(os.path.join(filepath, file), n_time_interval)

            batch_data_val = Data.DataLoader(data_val, batch_size=batch_size, drop_last=True)

            for id, batch in enumerate(batch_data_val):
                val_x, val_L, val_y, val_time_interval, val_rnn_index = batch

                model.eval()
                val_pred = model(val_x, val_L, n_steps,
                                 hidden_dim, val_rnn_index, val_time_interval)

                b_v_loss = criterion(val_pred, val_y)
                val_loss.append(b_v_loss)
                print('val_loss：', np.mean(val_loss))

        filepath = data_path + '\\data_test\\'
        filelist = os.listdir(filepath)
        for file in filelist:
            data_test = dataload.MyDataset(os.path.join(filepath, file), n_time_interval)

            batch_data_test = Data.DataLoader(data_test, batch_size=batch_size, drop_last=True)

            for id, batch in enumerate(batch_data_test):
                test_x, test_L, test_y, test_time_interval, test_rnn_index = batch

                model.eval()
                test_pred = model(test_x, test_L, n_steps,
                                  hidden_dim, test_rnn_index, test_time_interval)
                b_t_loss = criterion(test_pred, test_y)

                test_loss.append(b_t_loss)
                print('test_loss', np.mean(test_loss))

        if np.mean(val_loss) < best_val_loss:
            best_val_loss = np.mean(val_loss)
            best_test_loss = np.mean(test_loss)
            patience = max_try

        predict_result = []

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

        model.train()
        patience -= 1
        if not patience:
            break


print(len(predict_result), len(test_y))
print("Finished!\n----------------------------------------------------------------")
print("Time:", time.time() - start)
print("Valid Loss:", best_val_loss)
print("Test Loss:", best_test_loss)
