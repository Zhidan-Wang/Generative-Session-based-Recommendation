# 由all_train_seq.pickle生成real.data供seqgan使用
# 生产全部长度的样本当作数据
from collections import Counter
import pickle
file = open('data/diginetica/all_train_seq.pickle', 'rb')
data = pickle.load(file)

chose_data_len = {2: 50995, 3: 22214, 4: 13459, 5: 8038, 6: 5445, 7: 3597, 8: 2587, 9: 1944, 10: 1484, 11: 1170, 12: 872, 13: 686, 14: 592, 15: 441, 16: 336, 17: 296, 18: 246, 19: 215, 20: 183, 21: 152, 22: 138, 23: 136, 24: 114, 26: 84, 27: 72, 28: 63, 25: 61, 29: 53, 30: 40, 31: 37, 34: 33, 32: 31, 33: 29, 35: 28, 36: 23, 37: 21, 40: 16, 39: 15, 41: 13, 38: 13, 42: 12, 48: 11, 43: 11, 49: 9, 44: 9, 51: 8, 50: 8, 52: 7, 46: 7, 54: 7, 56: 6, 62: 6, 45: 6, 53: 6, 47: 6, 60: 6, 69: 5, 58: 5, 57: 5, 63: 4, 72: 4, 84: 3, 68: 3, 79: 3, 64: 3, 61: 3, 101: 2, 55: 2, 59: 2, 66: 2, 70: 2, 75: 2, 76: 2, 82: 2, 87: 2, 88: 1, 85: 1, 65: 1, 78: 1, 120: 1, 140: 1, 86: 1, 141: 1, 77: 1, 146: 1, 144: 1, 81: 1, 94: 1, 83: 1}
chose_data = {}
for seq in data:
    seq_len = len(seq)
    if seq_len not in chose_data.keys():
        # 第一次加入的序列,首先对字典进行初始化
        chose_data[seq_len] = []
    else:
        chose_data[seq_len].append(seq)
# 按照2,3,4,5,6的顺序写入文件
f = open('data/diginetica/real.data','w')
f.write('')
f.close()
f = open('data/diginetica/real.data','a')
chose_number = [2,3,4,5,6,7,8,9,10]
for number in chose_number:    # 对字典中每个列表进行遍历
    seq_set = chose_data[number] # seq列表
    set_number = int(chose_data_len[number]/64)*64
    print(set_number)
    for j in range(set_number):
        seq = seq_set[j]
        # 进行padding补齐
        pad_len = chose_number[-1] -len(seq)
        for i in range(0, pad_len):
            seq.append(0)
        for token in seq:
            f.write(str(token)+' ')
        f.write('\n')
f.close()
print('done')