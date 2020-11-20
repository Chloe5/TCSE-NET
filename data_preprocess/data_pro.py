# import pickle
# import math
# import gc
# path ="D:\\DKs-workshop\\canCN_pytorch\\dataset\\180_timeinterval"
# id_train, x_train, L, y_train, sz_train, time_train, vocabulary_size = pickle.load(open(path+'\\data_val.pkl', 'rb'))
# step =1
# filename = 'D:\\DKs-workshop\\canCN_pytorch\\dataset\\180_timeinterval\\data_val\\data_val_'
# print(len(id_train)) #train_41975 test_8950 val_8941
# for i in range(math.floor(len(id_train) / 320)):
#     pickle.dump((id_train[i * 320:(i + 1) * 320], x_train[i * 320:(i + 1) * 320], L[i * 320:(i + 1) * 320],
#                  y_train[i * 320:(i + 1) * 320],
#                  sz_train[i * 320:(i + 1) * 320], time_train[i * 320:(i + 1) * 320], vocabulary_size),
#                 open(filename + str(i) + '.pkl', 'wb'))
# if len(id_train) % 320 != 0:
#     pickle.dump((id_train[i * 320:], x_train[i * 320:], L[i * 320:],
#                  y_train[i * 320:],
#                  sz_train[i * 320:], time_train[i * 320:],
#                  vocabulary_size), open(filename + str(i + 1) + '.pkl', 'wb'))
#
# gc.collect()

path = 'D:\\DKs-workshop\\004.CasCN-master\\dataset_weibo'
filepath = path +'\\dataset_weibo.txt'
edges =set()
with open(filepath) as f:
    for line in f:
        line = line.split('\t')
        walk = line[4].split(' ')
        for w in walk:
            nodes =w.split(":")[0].split("/")
            for n in range(len(nodes)-1):
                if nodes[n] == nodes[n+1]:
                    continue
                edges.add((nodes[n],nodes[n+1]))


file = open(path + '\\networ_topology.txt','wb')
for edge in edges:
    e = str.encode(edge[0]+'\t'+edge[1]+'\n')
    file.write(e)