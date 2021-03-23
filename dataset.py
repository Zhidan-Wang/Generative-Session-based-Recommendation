import pickle
import numpy as np

from torch.utils.data import Dataset

def Preprocess(root, valid_portion=0.1, maxlen=19, sort_by_len=False):
    '''Loads the dataset

    :type path: String
    :param path: The path to the dataset (here RSC2015)
    :type n_items: int
    :param n_items: The number of items.
    :type valid_portion: float
    :param valid_portion: The proportion of the full train set used for
        the validation set.
    :type maxlen: None or positive int
    :param maxlen: the max sequence length we use in the train/valid set.
    :type sort_by_len: bool
    :name sort_by_len: Sort by the sequence lenght for the train,
        valid and test set. This allow faster execution as it cause
        less padding per minibatch. Another mechanism must be used to
        shuffle the train set at each epoch.

    '''

    # Load the dataset
    path_train_data = root + 'train.pickle'
    path_test_data = root + 'test.pickle'
    with open(path_train_data, 'rb') as f1:
        train_set = pickle.load(f1)

    with open(path_test_data, 'rb') as f2:
        test_set = pickle.load(f2)

    # 对超过指定长度的训练集和测试集序列,取最大长度
    if maxlen:
        new_train_set_x = []
        new_train_set_y = []
        for x, y in zip(train_set[0], train_set[1]):
            if len(x) < maxlen:
                new_train_set_x.append(x)
                new_train_set_y.append(y)
            else:
                new_train_set_x.append(x[:maxlen])
                new_train_set_y.append(y)
        train_set = (new_train_set_x, new_train_set_y)
        del new_train_set_x, new_train_set_y

        new_test_set_x = []
        new_test_set_y = []
        for xx, yy in zip(test_set[0], test_set[1]):
            if len(xx) < maxlen:
                new_test_set_x.append(xx)
                new_test_set_y.append(yy)
            else:
                new_test_set_x.append(xx[:maxlen])
                new_test_set_y.append(yy)
        test_set = (new_test_set_x, new_test_set_y)
        del new_test_set_x, new_test_set_y


    return train_set, test_set

class RecSysDataset(Dataset):
    """define the pytorch Dataset class for yoochoose and diginetica datasets.
    """
    def __init__(self, data):
        self.data = data
        print('-'*50)
        print('Dataset info:')
        print('Number of sessions: {}'.format(len(data[0])))
        print('-'*50)
        
    def __getitem__(self, index):
        session_items = self.data[0][index]
        target_item = self.data[1][index]
        return session_items, target_item

    def __len__(self):
        return len(self.data[0])

# add gan's data to train data
def add_data(train, merge=True):
    filename = './gen_data/gene.data'
    with open(filename, 'r+', encoding='utf-8') as f:
        gene_data = [i[:-1].split(' ') for i in f.readlines()]
    new_gene_data = []
    for cur_data in gene_data:
        cur_data = [x for x in cur_data if x != '0']
        new_gene_data.append(list(map(int, cur_data)))  
    if merge:  
        for line_data in new_gene_data:
            seq_len = len(line_data)
            for idx in range(1,seq_len):
                train[0].append( line_data[0:idx] )
                train[1].append( line_data[idx])
        return train
    else:
        new_train = [[],[]]
        for line_data in new_gene_data:
            seq_len = len(line_data)
            for idx in range(1,seq_len):
                new_train[0].append( line_data[0:idx] )
                new_train[1].append( line_data[idx])
        return new_train 