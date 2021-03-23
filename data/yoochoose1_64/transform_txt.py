# 将txt转换成pickle形式
import pickle
with open('data/yoochoose1_64/train.txt', 'rb')as file:
    data = pickle.load(file)
with open('data/yoochoose1_64/all_train.pickle', 'wb')as file:
    pickle.dump(data, file)

with open('data/yoochoose1_64/test.txt', 'rb')as file:
    data = pickle.load(file)
with open('data/yoochoose1_64/test.pickle', 'wb')as file:
    pickle.dump(data, file)

with open('data/yoochoose1_64/all_train_seq.txt', 'rb')as file:
    data = pickle.load(file)
with open('data/yoochoose1_64/all_train_seq.pickle', 'wb')as file:
    pickle.dump(data, file)


print('done')