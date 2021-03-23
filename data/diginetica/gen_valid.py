import pickle
import numpy as np

all_train_path = './data/diginetica/all_train.pickle'
with open(all_train_path, 'rb') as f:
    all_train_set = pickle.load(f)

valid_portion = 0.1
all_train_set_sample, all_train_set_target = all_train_set
n_samples = len(all_train_set_sample)
sidx = np.arange(n_samples, dtype='int32')
np.random.shuffle(sidx)
n_train = int(np.round(n_samples * (1. - valid_portion)))
valid_set_sample = [all_train_set_sample[s] for s in sidx[n_train:]]
valid_set_target = [all_train_set_target[s] for s in sidx[n_train:]]
train_set_sample = [all_train_set_sample[s] for s in sidx[:n_train]]
train_set_target = [all_train_set_target[s] for s in sidx[:n_train]]

# 汇总并保存
train_sample = (train_set_sample, train_set_target)
valid_sample = (valid_set_sample, valid_set_target)
with open('data/diginetica/train.pickle', 'wb')as file:
    pickle.dump(train_sample, file)
with open('data/diginetica/valid.pickle', 'wb')as file:
    pickle.dump(valid_sample, file)
print('done')