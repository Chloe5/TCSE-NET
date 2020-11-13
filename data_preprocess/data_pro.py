import pickle
import math
import gc
path ="D:\\学术相关\\GCN 项目\\dataset\\data_val"
id_train, x_train, L, y_train, sz_train, time_train, vocabulary_size = pickle.load(open(path+'\\data_val.pkl', 'rb'))
step =1
filename = 'D:\\学术相关\\GCN 项目\\dataset\\data_val\\data_val_'
print(len(id_train)) #train_41975 test_8950 val_8941
for i in range(math.floor(len(id_train) / 320)):
    pickle.dump((id_train[i * 320:(i + 1) * 320], x_train[i * 320:(i + 1) * 320], L[i * 320:(i + 1) * 320],
                 y_train[i * 320:(i + 1) * 320],
                 sz_train[i * 320:(i + 1) * 320], time_train[i * 320:(i + 1) * 320], vocabulary_size),
                open(filename + str(i) + '.pkl', 'wb'))
if len(id_train) % 320 != 0:
    pickle.dump((id_train[i * 320:], x_train[i * 320:], L[i * 320:],
                 y_train[i * 320:],
                 sz_train[i * 320:], time_train[i * 320:],
                 vocabulary_size), open(filename + str(i + 1) + '.pkl', 'wb'))

gc.collect()

