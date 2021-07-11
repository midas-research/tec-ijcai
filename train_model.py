import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
import os
import math
import glob
import time
import random
import numpy as np
import torch.optim as optim
import pickle
from torch.autograd import Variable
from torch_geometric import utils
from tqdm import tqdm
import torch.optim as optim 
import networkx as nx
from sklearn.metrics import accuracy_score, confusion_matrix

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

class Attention(nn.Module):


    def __init__(self, dimensions, attention_type='general'):
        super(Attention, self).__init__()

        if attention_type not in ['dot', 'general']:
            raise ValueError('Invalid attention type selected.')

        self.attention_type = attention_type
        if self.attention_type == 'general':
            self.linear_in = nn.Linear(dimensions, dimensions, bias=False)

        self.linear_out = nn.Linear(dimensions * 2, dimensions, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()

    def forward(self, query, context):

        batch_size, output_len, dimensions = query.size()
        query_len = context.size(1)

        if self.attention_type == "general":
            query = query.reshape(batch_size * output_len, dimensions)
            query = self.linear_in(query)
            query = query.reshape(batch_size, output_len, dimensions)

        attention_scores = torch.bmm(query, context.transpose(1, 2).contiguous())

        # Compute weights across every context sequence
        attention_scores = attention_scores.view(batch_size * output_len, query_len)
        attention_weights = self.softmax(attention_scores)
        attention_weights = attention_weights.view(batch_size, output_len, query_len)

        mix = torch.bmm(attention_weights, context)


        combined = torch.cat((mix, query), dim=2)
        combined = combined.view(batch_size * output_len, 2 * dimensions)

        output = self.linear_out(combined).view(batch_size, output_len, dimensions)
        output = self.tanh(output)

        return output, attention_weights

class gru(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(gru, self).__init__()
        self.gru1 = torch.nn.GRU(input_size = input_size, hidden_size=hidden_size, batch_first=False)
    def forward(self, inputs):
        full, last  = self.gru1(inputs)
        return full,last


class ASTGPOLS(nn.Module):

    def __init__(self, nfeat, nhid, nclass, dropout, nheads, lookback):
        super(ASTGPOLS, self).__init__()
        self.nhid = nhid
        self.gat = GATConv(nfeat, nhid, heads=nheads, negative_slope=0.2,concat=False, dropout=dropout)
        self.attention = Attention(nhid)
        self.gru1 = gru(nhid,nhid)
        self.linear = nn.Linear(nhid,nclass)
        self.time_steps = lookback
    def forward(self, graph, fts):
        y_full = []
        for i in range(self.time_steps+1):
            x = fts[i]
            G = graph[i]
            y = F.leaky_relu(self.gat(x,G),0.2)
            y_full.append(y.reshape(1,x.shape[0],self.nhid))
        y = torch.cat(y_full)
        context,query = self.gru1(y)
        query = query.permute(1,0,2)
        context = context.permute(1,0,2)
        output, weights = self.attention(query, context)
        output = F.leaky_relu(output.reshape((x.shape[0],self.nhid)),0.2)
        output = self.linear(output)
        return output


path_save_graph = "/content/graphs/" 
path_save_numpy = "/content/node_fts/"
path_save_label =  "/content/labels/"
path_save_mask =  "/content/masks/"

loss_funct = torch.nn.CrossEntropyLoss()

def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    acc_train = []
    net_loss_train = 0
    for i in range(1396): #
        start = time.time()
        train_features = torch.FloatTensor(np.load(path_save_numpy + str(i)+'.npy')).to('cuda')
        train_mask = pickle.load( open( path_save_mask+str(i)+".p", "rb" ))
        train_label = torch.LongTensor(np.load(path_save_label + str(i)+'.npy')).to('cuda')
        all_graphs = pickle.load( open( path_save_graph+str(i)+".p", "rb" ) )
        all_graphs = [utils.from_scipy_sparse_matrix(nx.to_scipy_sparse_matrix(G))[0].to('cuda') for G in all_graphs]
        end = time.time()
        compute = end-start
        start = time.time()
        output = model(all_graphs, train_features)
        loss_train = loss_funct(output[train_mask], train_label)
        acc_train.append(accuracy(output[train_mask], train_label).detach().cpu().numpy())
        net_loss_train = net_loss_train+loss_train.detach().cpu().numpy()
        loss_train.backward()
        optimizer.step()
        end = time.time()
    model.eval()
    accuracy_val = []
    accuracy_test = []
    for i in range(1396,1596): #
        train_features = torch.FloatTensor(np.load(path_save_numpy + str(i)+'.npy')).to('cuda')
        train_mask = pickle.load( open( path_save_mask+str(i)+".p", "rb" ))
        train_label = torch.LongTensor(np.load(path_save_label + str(i)+'.npy')).to('cuda')
        all_graphs = pickle.load( open( path_save_graph+str(i)+".p", "rb" ) )
        all_graphs = [utils.from_scipy_sparse_matrix(nx.to_scipy_sparse_matrix(G))[0].to('cuda') for G in all_graphs]
        output = model(all_graphs, train_features)
        loss_train = F.nll_loss(output[train_mask], train_label)

        accuracy_val.append(accuracy(output[train_mask], train_label).detach().cpu().numpy())
    # oli = []
    # lli = []
    tc = np.zeros((2,2))
    for i in range(1596, 1995): #
        train_features = torch.FloatTensor(np.load(path_save_numpy + str(i)+'.npy')).to('cuda')
        train_mask = pickle.load( open( path_save_mask+str(i)+".p", "rb" ))
        train_label = torch.LongTensor(np.load(path_save_label + str(i)+'.npy')).to('cuda')
        all_graphs = pickle.load( open( path_save_graph+str(i)+".p", "rb" ) )
        all_graphs = [utils.from_scipy_sparse_matrix(nx.to_scipy_sparse_matrix(G))[0].to('cuda') for G in all_graphs]
        output = model(all_graphs, train_features)
        loss_train = F.nll_loss(output[train_mask], train_label)
        tc = tc + np.array(confusion_matrix(train_label.detach().cpu().numpy(),np.argmax(output[train_mask].detach().cpu().numpy(),-1)))
        accuracy_test.append(accuracy(output[train_mask], train_label).detach().cpu().numpy())
    tn = tc[0,0]
    fn = tc[1,0]
    tp = tc[1,1]
    fp = tc[0,1]
    tacc = (tp+tn)/(tp+tn+fp+fn)
    tf1 = (tp)/(tp+0.5*(fp+fn))
    tmcc = (tp*tn-fp*fn)/(math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))

epochs = 100
l_r = 1e-4
weight_decay = 5e-4
model = ASTGPOLS(nfeat=768, nhid=16, nclass=2, dropout=0.3, nheads=4, lookback=15).to('cuda')
optimizer = optim.Adam(model.parameters(), 
                       lr=l_r, 
                       weight_decay=weight_decay)

checkpoint = torch.load('6months')
for i in tqdm(range(epochs)):
    train(i)
    torch.save({
            'epoch': epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
            }, '6months')

