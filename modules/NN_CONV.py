import torch
from torch.nn import Linear
from torch.nn import Dropout
from torch_geometric.nn import NNConv 


class NNCONV(torch.nn.Module):
    def __init__(self, embedding_size, dropout):
        super().__init__()
        self.embed_nnet = torch.nn.Sequential(Linear(5,10), torch.nn.ReLU(), Linear(10, 10*embedding_size))
        self.main_nnet = torch.nn.Sequential(Linear(5,10), torch.nn.ReLU(), Linear(10, embedding_size*embedding_size))
        self.end_nnet =  torch.nn.Sequential(Linear(5,10), torch.nn.ReLU(), Linear(10, embedding_size*10))

        self.embed_layer = NNConv(10, embedding_size, self.embed_nnet, aggr='mean')
        self.layer1 = NNConv(embedding_size, embedding_size, self.main_nnet, aggr='mean')
        self.layer2 = NNConv(embedding_size, embedding_size, self.main_nnet, aggr='mean')
        self.layer3 = NNConv(embedding_size, embedding_size, self.main_nnet, aggr='mean')
        self.layer4 = NNConv(embedding_size, embedding_size, self.main_nnet, aggr='mean')
        self.layer5 = NNConv(embedding_size, embedding_size, self.main_nnet, aggr='mean')
        self.layer6 = NNConv(embedding_size, embedding_size, self.main_nnet, aggr='mean')
        self.end_layer = NNConv(embedding_size, 10, self.end_nnet, aggr='mean')
        
        self.dropout=Dropout(p=dropout)
        
        self.out = Linear(10, 1)
        
    def forward(self, num_layers, x, edge_index, edge_attr):
        hidden = self.embed_layer(x, edge_index, edge_attr)
        hidden = hidden.relu()
        
        hidden = self.dropout(hidden)
        
        if num_layers == 1:
            hidden = self.layer1(hidden, edge_index, edge_attr)
            hidden = hidden.relu()
            
        elif num_layers == 2:
            hidden = self.layer1(hidden, edge_index, edge_attr)
            hidden = hidden.relu()

            hidden = self.layer2(hidden, edge_index, edge_attr)
            hidden = hidden.relu()

        elif num_layers == 3:
            hidden = self.layer1(hidden, edge_index, edge_attr)
            hidden = hidden.relu()

            hidden = self.layer2(hidden, edge_index, edge_attr)
            hidden = hidden.relu()
        
            hidden = self.layer3(hidden, edge_index, edge_attr)
            hidden = hidden.relu()

        elif num_layers == 4:
            hidden = self.layer1(hidden, edge_index, edge_attr)
            hidden = hidden.relu()

            hidden = self.layer2(hidden, edge_index, edge_attr)
            hidden = hidden.relu()

            hidden = self.layer3(hidden, edge_index, edge_attr)
            hidden = hidden.relu()

            hidden = self.layer4(hidden, edge_index, edge_attr)
            hidden = hidden.relu()

        elif num_layers == 5:
            hidden = self.layer1(hidden, edge_index, edge_attr)
            hidden = hidden.relu()

            hidden = self.layer2(hidden, edge_index, edge_attr)
            hidden = hidden.relu()

            hidden = self.layer3(hidden, edge_index, edge_attr)
            hidden = hidden.relu()

            hidden = self.layer4(hidden, edge_index, edge_attr)
            hidden = hidden.relu()

            hidden = self.layer5(hidden, edge_index, edge_attr)
            hidden = hidden.relu()

        elif num_layers == 6:
            hidden = self.layer1(hidden, edge_index, edge_attr)
            hidden = hidden.relu()

            hidden = self.layer2(hidden, edge_index, edge_attr)
            hidden = hidden.relu()

            hidden = self.layer3(hidden, edge_index, edge_attr)
            hidden = hidden.relu()

            hidden = self.layer4(hidden, edge_index, edge_attr)
            hidden = hidden.relu()

            hidden = self.layer5(hidden, edge_index, edge_attr)
            hidden = hidden.relu()

            hidden = self.layer6(hidden, edge_index, edge_attr)
            hidden = hidden.relu()

        hidden = self.end_layer(hidden, edge_index, edge_attr)
        hidden = hidden.relu()
        
        out = self.out(hidden)
        
        return out
