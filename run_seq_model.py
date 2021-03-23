import metric

import time
import random
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.optim as optim
import argparse

from gru4rec import GRU4Rec
from narm import NARM
from sasrec import SASRec
from stamp import STAMP
from dataset import *
from untils import rec15_collate_fn, save_model
from untils import get_config

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch import nn



def trainForEpoch(train_loader, model, optimizer):
    model.train()

    sum_epoch_loss = 0

    for seq, target, lens in train_loader:
        seq = seq.to(device)
        target = target.to(device)
        lens = torch.tensor(lens).to(device)
        optimizer.zero_grad()
        loss = model.calculate_loss(seq, lens, target)
        loss.backward()
        optimizer.step() 

def validate(valid_loader, model):
    model.eval()
    recalls = []
    mrrs = []
    with torch.no_grad():
        for seq, target, lens in (valid_loader):
            seq = seq.to(device)
            target = target.to(device)
            lens = torch.tensor(lens).to(device)
            scores = model.full_sort_predict(seq, lens)
            logits = F.softmax(scores, dim = 1)
            recall, mrr = metric.evaluate(logits, target, k = topk)
            recalls.append(recall)
            mrrs.append(mrr)
    
    mean_recall = np.mean(recalls)
    mean_mrr = np.mean(mrrs)
    return mean_recall, mean_mrr

# ----------------------------------------init config----------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='gru4rec') # gru4rec narm searec stamp
parser.add_argument('--dataset_name', type=str, default='yoochoose1_64') # 'yoochoose1_64' 'diginetica'
parser.add_argument('--train_mode', type=str, default='ori') # ori, gan
args = parser.parse_args()
model_name = args.model_name
dataset_name = args.dataset_name
train_mode = args.train_mode

    
# ----------------------------------------init seed----------------------------------------
SEED = 2020
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True

# ----------------------------------------init parameters and model----------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if model_name == 'gru4rec':
    batch_size = 32
    epoch_number = 20
    lr = 0.001
    lr_dc = 0.1
    lr_dc_step = 80
    topk = 20
    config = get_config(model_name, dataset_name)
    model = GRU4Rec(config).to(device)
elif model_name == 'narm':
    batch_size = 512
    epoch_number = 100
    lr = 0.001
    lr_dc = 0.1
    lr_dc_step = 80
    topk = 20
    config = get_config(model_name, dataset_name)
    model = NARM(config).to(device)
elif model_name == 'sasrec':
    batch_size = 32
    epoch_number = 20
    lr = 0.001
    lr_dc = 0.1
    lr_dc_step = 80
    topk = 20
    config = get_config(model_name, dataset_name)
    model = SASRec(config).to(device)
elif model_name == 'stamp':
    batch_size = 512
    epoch_number = 30
    lr = 0.001
    lr_dc = 0.1
    lr_dc_step = 80
    topk = 20
    config = get_config(model_name, dataset_name)
    model = STAMP(config).to(device)

# ----------------------------------------init model----------------------------------------
optimizer = optim.Adam(model.parameters(), lr)
scheduler = StepLR(optimizer, step_size = lr_dc_step, gamma = lr_dc)

# ----------------------------------------load data----------------------------------------
if dataset_name == 'yoochoose1_64':
    data_root = './data/yoochoose1_64/'
elif dataset_name == 'diginetica':
    data_root = './data/diginetica/'

train, test = Preprocess(data_root)
train_dataset = RecSysDataset(train)
test_dataset = RecSysDataset(test)
train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, collate_fn = rec15_collate_fn)
test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False, collate_fn = rec15_collate_fn)

# ----------------------------------------train----------------------------------------
log_filename = './log/' + model_name + '_' + dataset_name + '_' + train_mode + '.log'
with open(log_filename, 'w') as f:
    for epoch in (range(epoch_number)):
        start_time = time.time()
        # train
        trainForEpoch(train_loader, model, optimizer)
        scheduler.step()
        # eval
        recall, mrr = validate(test_loader, model)
        print('Epoch {} : Recall@{}: {:.4f}, MRR@{} : {:.4f} time:{}\n'.format(epoch, topk, recall, topk, mrr, int(time.time()-start_time)))
        log = 'Epoch {} : Recall@{}: {:.4f}, MRR@{} : {:.4f} time:{}\n'.format(epoch, topk, recall, topk, mrr, int(time.time()-start_time))
        f.write(log)
        # save model
        save_filename = model_name + '_' + dataset_name + '_' + train_mode
        save_model(model, save_filename)
