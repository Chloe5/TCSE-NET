import math
DATA_PATHA = "D:\\DKs-workshop\\004.CasCN-master\\dataset_weibo"


cascades  = DATA_PATHA+"\\dataset_weibo.txt"

cascade_train = DATA_PATHA+"\\cascade_train.txt"
cascade_val = DATA_PATHA+"\\cascade_val.txt"
cascade_test = DATA_PATHA+"\\cascade_test.txt"
shortestpath_train = DATA_PATHA+"\\shortestpath_train.txt"
shortestpath_val = DATA_PATHA+"\\shortestpath_val.txt"
shortestpath_test = DATA_PATHA+"\\shortestpath_test.txt"

train_pkl = "D:\\DKs-workshop\\canCN_pytorch\\dataset\\180_timeinterval\\data_train.pkl"
val_pkl = "D:\\DKs-workshop\\canCN_pytorch\\dataset\\180_timeinterval\\data_val.pkl"
test_pkl = "D:\\DKs-workshop\\canCN_pytorch\\dataset\\180_timeinterval\\data_test.pkl"
information = "D:\\DKs-workshop\\canCN_pytorch\\dataset\\180_timeinterval\\information.pkl"

# parameters
observation = 3 * 60 * 60 - 1
print("observation time", observation)
n_time_interval = 6
print("the number of time interval:", n_time_interval)
time_interval = math.ceil((observation + 1) * 1.0 / n_time_interval)  # 向上取整
print("time interval:", time_interval)
lmax = 2

observation = 3*60*60-1 # 前三小时作为数如的snapshot
pre_times = [24 * 3600] #观测至24小时，即最终预测为第24小时的扩散级联size