from collections import Counter
import pickle
file = open('data/diginetica/all_train_seq.pickle', 'rb')
data = pickle.load(file)

# 统计seq长度分布
len_set = []
for seq in data:
    len_set.append(len(seq))
chose_data_len = Counter(len_set)
print(chose_data_len)