import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math

criterion = nn.CrossEntropyLoss()
batch_size = 5
emb_dim = 10
hidden_dim = 8
num_emb = 6
lstm = nn.GRU(emb_dim, hidden_dim, batch_first=True)
emb = nn.Embedding(num_emb, emb_dim, padding_idx=0)
softmax = nn.LogSoftmax(dim=1)
lin = nn.Linear(hidden_dim, num_emb)

def init_hidden(batch_size):
    h = Variable(torch.zeros((1, batch_size, hidden_dim)))
    #c = Variable(torch.zeros((1, batch_size, self.hidden_dim)))
    #h, c = h.to(self.device), c.to(self.device)
    return h

def step(x, h):
    emb = nn.Embedding(num_emb, emb_dim, padding_idx=0)
    x_emb = emb(x)
    #print("x_emb",x_emb.size())
    output, h = lstm(x_emb, h)
    #print("output", output.size())
    print("h",h.size())
    #print(output[:][-1][:])
    #print(h)
    pred = F.softmax(lin(output.view(-1, hidden_dim)), dim=1)
    pred1 = F.softmax(lin(h.view(-1, hidden_dim)), dim=1)
    print("pred",pred) # [batch, n_items]
    print("pred1", pred1)
    return pred, h

x = Variable(torch.zeros((batch_size, 1)).long())
h = init_hidden(batch_size)
#print(x)
#print("h",h.size())
samples = []
for i in range(10):
    output, h = step(x, h)
    x = output.multinomial(1)
    #print("x", x)
    samples.append(x)

pred = output
tar = samples[-1].squeeze(1)
target = torch.zeros(1).long()
print("#",target.size())
print(output.size())
loss = criterion(pred, tar)
print(loss)
#print(len(samples))

#print(len(samples[0]))

