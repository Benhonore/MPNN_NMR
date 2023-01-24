import torch
from torch.nn import Linear
from torch.nn import Dropout
from torch_geometric.nn import NNConv 


class NNConv(torch.nn.Module):
    def __init__(self, embedding_size, dropout):
        super().__init__()
        self.nnet = torch.nn.Sequential(Linear(2,10), torch.nn.ReLU(), Linear(10, 1*embedding_size))
        self.nnet2 = torch.nn.Sequential(Linear(2,10), torch.nn.ReLU(), Linear(10, embedding_size*embedding_size))
        self.nnet3 = torch.nn.Sequential(Linear(2,10), torch.nn.ReLU(), Linear(10, embedding_size*embedding_size))
        self.nnet4 =  torch.nn.Sequential(Linear(2,10), torch.nn.ReLU(), Linear(10, embedding_size*10))
        #self.nnet4 =  torch.nn.Sequential(Linear(1,10), torch.nn.ReLU(), Linear(10, 10))

        self.layer = NNConv(1, embedding_size, self.nnet, aggr='mean')
        self.layer2 = NNConv(embedding_size, embedding_size, self.nnet2, aggr='mean')
        self.layer3 = NNConv(embedding_size, embedding_size, self.nnet3, aggr='mean')
        self.layer4 = NNConv(embedding_size, 10, self.nnet4, aggr='mean')
        #self.layer4 = NNConv(10, 1, self.nnet4, aggr='mean')
        
        self.dropout=Dropout(p=dropout)
        
        self.out = Linear(10, 1)
        
    def forward(self, x, edge_index, edge_attr):
        hidden = self.layer(x, edge_index, edge_attr)
        hidden = hidden.relu()
        
        hidden = self.dropout(hidden)
        
        hidden = self.layer2(hidden, edge_index, edge_attr)
        hidden = hidden.relu()
        
        hidden = self.layer3(hidden, edge_index, edge_attr)
        hidden = hidden.relu()
        
        hidden = self.layer4(hidden, edge_index, edge_attr)
        hidden = hidden.relu()
        
        #hidden = self.conv3(hidden, edge_index, edge_attr)
        #out = hidden.relu()
        
        out = self.out(hidden)
        
        return out
