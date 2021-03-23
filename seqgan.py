import os
import random
import math
import argparse
import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from gan.generator import Generator
from gan.discriminator import Discriminator
from gan.target_lstm import TargetLSTM
from gan.rollout import Rollout
from gan.data_iter import GenDataIter, DisDataIter
from narm import NARM
from gru4rec import GRU4Rec
from sasrec import SASRec
from stamp import STAMP
from untils import get_config

# ---------------------------------------init config---------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='gru4rec') # gru4rec narm searec stamp
parser.add_argument('--dataset_name', type=str, default='yoochoose1_64') # 'yoochoose1_64' 'diginetica'
parser.add_argument('--TOTAL_BATCH', type=int, default=100) # 'yoochoose1_64' 'diginetica'
args = parser.parse_args()
model_name = args.model_name
dataset_name = args.dataset_name
TOTAL_BATCH = args.TOTAL_BATCH
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_model():
    config = get_config(model_name, dataset_name)
    if model_name == 'narm':
        model = NARM(config).to(device)
    elif model_name == 'gru4rec':
        model = GRU4Rec(config).to(device)
    elif model_name == 'sasrec':
        model = SASRec(config).to(device)
    elif model_name == 'stamp':
        model = STAMP(config).to(device)
    saved_model_name = './saved_model/' + model_name+'_'+dataset_name+'_'+'ori'+'.pickle'
    state_dict = torch.load(saved_model_name, map_location=device)
    model.load_state_dict(state_dict)
    return model

def get_loss(model, prob, targets, samples):
    """
            Args:
                prob: (N, C), torch Variable (N = seq_num * seq_len)
                targets : (N, ), torch Variable
                samples : (seq_num , seq_len)
                reward : (N, ), torch Variable
    """
    rewards = []
    model.eval()
    # 传入序列以及序列对应的长度,以及
    choose_bool = []
    for i, single_sample in enumerate(samples):  #seq_num
        for j, word in enumerate(single_sample): #seq_len
            if j!=0 and word!=0:
                # 添加新的元素
                seq = single_sample[0:j]
                seq_len = [len(seq)]
                target = torch.zeros(1).long()
                target[0] = single_sample[j]  #(1,)
                seq = seq.to(device)
                seq = seq.unsqueeze(0) # (1, j)
                target = target.to(device)
                seq_len = torch.tensor(seq_len).to(device)
                loss = model.calculate_loss(seq, seq_len, target)
                rewards.append(loss.item())
                choose_bool.append([True]*VOCAB_SIZE)  # C
            else:
                choose_bool.append([False]*VOCAB_SIZE)
    choose_bool = torch.tensor(choose_bool).to(device)
    # gen one_hot
    N = targets.size(0)
    C = prob.size(1)
    one_hot = torch.zeros((N, C))
    one_hot = one_hot.to(device)
    one_hot.scatter_(1, targets.data.view((-1,1)), 1)
    one_hot = one_hot.bool()
    # change rewards when rewards=0
    rewards = [ 1e-7 if  single_reward<=1e-7 else single_reward for single_reward in rewards]
    rewards = torch.log(torch.tensor(rewards)).to(device)
    # cal loss
    loss = torch.masked_select(prob, one_hot&choose_bool)    #[seq_num*seq_len - num_0, 1]
    loss = rewards*loss
    loss = -torch.mean(loss)
    return loss

def generate_samples(model, batch_size, generated_num, output_file):
    samples = []
    for _ in range(int(generated_num / batch_size)):
        sample = model.sample(batch_size, g_sequence_len).cpu().data.numpy().tolist() #[batch, 10]
        samples.extend(sample) #[generated_num, 10]
    with open(output_file, 'w') as fout:
        for sample in samples:
            string = ' '.join([str(s) for s in sample])
            fout.write('%s\n' % string)

def train_epoch(model, data_iter, criterion, optimizer):
    total_loss = 0.
    total_words = 0.
    for (data, target) in data_iter:#tqdm(
        #data_iter, mininterval=2, desc=' - Training', leave=False):
        data = Variable(data)
        target = Variable(target)
        data, target = data.to(device), target.to(device)
        target = target.contiguous().view(-1)
        pred = model.forward(data)
        loss = criterion(pred, target)
        total_loss += loss.item()
        total_words += data.size(0) * data.size(1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    data_iter.reset()
    return math.exp(total_loss / total_words)


# GAN的损失函数
class GANLoss(nn.Module):
    """Reward-Refined NLLLoss Function for adversial training of Gnerator"""
    def __init__(self):
        super(GANLoss, self).__init__()

    def forward(self, prob, targets, reward):
        """
        Args:
            prob: (N, C), torch Variable
            targets : (N, ), torch Variable
            reward : (N, ), torch Variable
        """
        N = targets.size(0)
        C = prob.size(1)
        one_hot = torch.zeros((N, C))
        if prob.is_cuda:
            one_hot = one_hot.cuda()
        one_hot.scatter_(1, targets.data.view((-1,1)), 1)
        one_hot = one_hot.bool()
        # one_hot = Variable(one_hot)
        # if prob.is_cuda:
        #     one_hot = one_hot.cuda()
        loss = torch.masked_select(prob, one_hot) #（N, ）
        loss = loss * reward
        loss = -torch.mean(loss)
        return loss

# init paramaters 
SEED = 88   # 随机种子
BATCH_SIZE = 512 # 一batch的数据的量
GENERATED_NUM = 5000
POSITIVE_FILE = './gen_data/real.data' # 真实数据的文件路径
POSITIVE_FILE = './data/' + dataset_name+ '/real.data' 
NEGATIVE_FILE = './gan_data/'+model_name+'_'+dataset_name+'.data'
if dataset_name == 'yoochoose1_64':
    VOCAB_SIZE = 37485
elif dataset_name == 'diginetica':
    VOCAB_SIZE = 43098
PRE_EPOCH_NUM = 120

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
g_emb_dim = 32
g_hidden_dim = 32
g_sequence_len = 10
d_emb_dim = 64
d_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
d_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 160, 160]
d_dropout = 0.75
d_num_class = 2

# init seed
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True

# init model
generator = Generator(VOCAB_SIZE, g_emb_dim, g_hidden_dim, device)
discriminator = Discriminator(d_num_class, VOCAB_SIZE, d_emb_dim, d_filter_sizes, d_num_filters, d_dropout)
generator = generator.to(device)
discriminator = discriminator.to(device)
seq_model = get_model()
seq_model = seq_model.to(device)
loss_set1 = []
loss_set2 = []

# ------------------------------------------train G and D------------------------------------------
gen_data_iter = GenDataIter(POSITIVE_FILE, BATCH_SIZE)
gen_criterion = nn.NLLLoss(reduction='sum')
gen_optimizer = optim.Adam(generator.parameters())
gen_criterion = gen_criterion.to(device)
print('Pretrain Generator ...')
for epoch in range(PRE_EPOCH_NUM):
    loss = train_epoch(generator, gen_data_iter, gen_criterion, gen_optimizer)
    print('Epoch [%d] G Model Loss: %f'% (epoch, loss))
dis_criterion = nn.NLLLoss(reduction='sum')             # 判别器损失函数
dis_optimizer = optim.Adam(discriminator.parameters())  # 判别器优化算法
dis_criterion = dis_criterion.to(device)
print('Pretrain Discriminator ...')
for epoch in range(5):
    generate_samples(generator, BATCH_SIZE, GENERATED_NUM, NEGATIVE_FILE)       # 此处是采用生成器生成虚假的数据： 5000
    dis_data_iter = DisDataIter(POSITIVE_FILE, NEGATIVE_FILE, BATCH_SIZE)       # 虚假的数据用另一个数据迭代器
    for _ in range(3):
        loss = train_epoch(discriminator, dis_data_iter, dis_criterion, dis_optimizer)
        print('Epoch [%d], D Model loss: %f' % (epoch, loss))
# # save model
# G_state_dict = generator.state_dict()
# torch.save(G_state_dict, 'generator.pickle')
# D_state_dict = discriminator.state_dict()
# torch.save(D_state_dict, 'discriminator.pickle')

# # ------------------------------------------load pre-train G and D------------------------------------------
# generator.load_state_dict(torch.load('./gan_model/G.pickle', map_location=device))
# discriminator.load_state_dict(torch.load('./gan_model/D.pickle', map_location=device))

rollout = Rollout(generator, 0.8)
print('Start Adeversatial Training...\n')
gen_gan_loss = GANLoss()
gen_gan_loss = gen_gan_loss.to(device)
gen_gan_optm = optim.Adam(generator.parameters()) 
gen_criterion = nn.NLLLoss(reduction='sum')
gen_criterion = gen_criterion.to(device)
dis_optimizer = optim.Adam(discriminator.parameters())     
dis_criterion = nn.NLLLoss(reduction='sum')    
dis_criterion = dis_criterion.to(device)
for total_batch in range(TOTAL_BATCH):
    ## Train the generator for one step
    for it in range(2):
        samples = generator.sample(BATCH_SIZE, g_sequence_len)
        # construct the input to the generator, add zeros before samples and delete the last column
        zeros = torch.zeros((BATCH_SIZE, 1)).type(torch.LongTensor)
        if samples.is_cuda:
            zeros = zeros.cuda()
        inputs = Variable(torch.cat([zeros, samples.data], dim = 1)[:, :-1].contiguous())   #batch, seq_len
        targets = Variable(samples.data).contiguous().view((-1,))
        prob = generator.forward(inputs)
        rewards1 = rollout.get_discriminator_reward(samples, 16, discriminator)
        rewards1 = Variable(torch.Tensor(rewards1))# -0.6590
        rewards1 = torch.exp(rewards1)
        rewards1 = rewards1.contiguous().view((-1,)) #(batch * seq_len, )
        rewards1 = rewards1.to(device)
        loss1 = gen_gan_loss(prob, targets, rewards1)
        loss2 = get_loss(seq_model, prob, targets, samples)
        loss1 = loss1.to(device)
        loss2 = loss2.to(device)
        loss = loss1+loss2
        gen_gan_optm.zero_grad()
        loss.backward()
        gen_gan_optm.step()
    rollout.update_params()
    print('Batch [%d] True Loss: %f' % (total_batch, loss.item()))
    print(loss1.item(), loss2.item())
    loss_set1.append(loss1.item())
    loss_set2.append(loss2.item())
    
    for _ in range(2):
        generate_samples(generator, BATCH_SIZE, GENERATED_NUM, NEGATIVE_FILE)
        dis_data_iter = DisDataIter(POSITIVE_FILE, NEGATIVE_FILE, BATCH_SIZE)
        for _ in range(2):
            loss = train_epoch(discriminator, dis_data_iter, dis_criterion, dis_optimizer)
            print('D Model loss : %f' % (loss))
generate_samples(generator, BATCH_SIZE, 10000, NEGATIVE_FILE) 

G_state_dict = generator.state_dict()

torch.save(G_state_dict, './gan_model/'+model_name+'_'+dataset_name+'G.pickle')
D_state_dict = discriminator.state_dict()
torch.save(D_state_dict, './gan_model/'+model_name+'_'+dataset_name+'D.pickle')
loss1_filename = './gan_data/' + model_name+'_'+dataset_name+'_loss1.npy'
loss2_filename = './gan_data/' + model_name+'_'+dataset_name+'_loss2.npy'
np.save(loss1_filename, loss_set1)
np.save(loss2_filename, loss_set2)