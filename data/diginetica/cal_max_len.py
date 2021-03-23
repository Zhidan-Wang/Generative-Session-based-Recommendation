# 计算训练集中最长的序列长度
import pickle

with open('data/diginetica/train.pickle', 'rb')as file:
    data = pickle.load(file)
sample, target = data
max_len = -1
for single_sample in sample:
    if len(single_sample)>max_len:
        max_len = len(single_sample)
print(max_len)
# 69