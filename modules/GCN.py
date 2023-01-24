import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.nn import Linear
from torch.nn import Dropout


class GCN(torch.nn.Module):
    def __init__(self, in_features, embedding_size, dropout):
        super().__init__()
      
        self.initial_conv = GCNConv(in_features, embedding_size)      
        
        self.conv1 = GCNConv(embedding_size, embedding_size)
        self.conv2 = GCNConv(embedding_size, embedding_size)
        self.conv3 = GCNConv(embedding_size, embedding_size)

        self.dropout = Dropout(p=dropout)

        self.out= Linear(embedding_size, 1)
                   
                    
    def forward(self, x, edge_index):
        
        hidden = F.relu(self.initial_conv(x, edge_index), inplace=True)
        hidden = self.dropout(hidden)

        hidden = F.relu(self.conv1(hidden, edge_index))
        hidden = F.relu(self.conv2(hidden, edge_index))
        hidden = F.relu(self.conv3(hidden, edge_index))
        
        out = self.out(hidden)

        return out
