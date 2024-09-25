# pytorch lib
import torch
import torch.nn as nn
import dgl.nn as dglnn

# Define a GNN
class SAGE(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, dropout_rate, hid_layer):
        super().__init__()
        self.conv1 = dglnn.SAGEConv(in_feats, hid_feats, aggregator_type='pool')
        self.conv2 = dglnn.SAGEConv(hid_feats, hid_feats, aggregator_type='pool')
        self.conv3 = dglnn.SAGEConv(hid_feats, out_feats, aggregator_type='pool')
        self.norm = nn.BatchNorm1d(hid_feats)
        self.activation = torch.nn.Tanh()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.hid_layer = hid_layer
    def forward(self, graph, inputs):
        h = self.conv1(graph, inputs) 
        # h = self.norm(h)
        h = self.activation(h)
        h = self.dropout(h)
        for i in range(self.hid_layer):
            h = self.conv2(graph, h)
            # h = self.norm(h)
            h = self.activation(h)
            h = self.dropout(h)
        h = self.conv3(graph, h)
        return h

class GIN(nn.Module):
    def __init__(self, in_feats, out_feats):
        super().__init__()
        self.conv1 = dglnn.GINConv(nn.Linear(in_feats, out_feats))
    def forward(self, graph, node_inputs):
        h = self.conv1(graph, node_inputs)
        return h

class ChebConv(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, dropout_rate, hid_layer):
        super().__init__()
        self.conv1 = dglnn.ChebConv(in_feats, hid_feats, 2)
        self.conv2 = dglnn.ChebConv(hid_feats, hid_feats, 2)
        self.conv3 = dglnn.ChebConv(hid_feats, out_feats, 2)
        self.activation = torch.nn.Tanh()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.hid_layer = hid_layer
    def forward(self, graph, inputs):
        h = self.conv1(graph, inputs) 
        h = self.activation(h)
        # h = self.dropout(h)
        for i in range(self.hid_layer):
            h = self.conv2(graph, h)
            h = self.activation(h)
            # h = self.dropout(h)
        h = self.conv3(graph, h)
        return h

# Define a MLP
class MLP(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout_rate, hid_layer):
        super().__init__()
        self.layer1 = nn.Linear(in_dim, hid_dim)
        self.layer2 = nn.Linear(hid_dim, hid_dim)
        self.layer3 = nn.Linear(hid_dim, out_dim)
        self.activation = torch.nn.Tanh()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.hid_layer = hid_layer
    def forward(self, x):
        y = self.layer1(x)
        y = self.activation(y)
        y = self.dropout(y)
        for i in range(self.hid_layer):
            y = self.layer2( y)
            y = self.activation(y)
            y = self.dropout(y)
        y = self.layer3(y)
        return y


# Define the entire model
class Model_SAGE(nn.Module):   # GraphSAGE
    def __init__(self, dim, time_L, fft_n, mode_N, dropout_rate, hid_layer):
        super().__init__()
        # use built-in Transformer ##################################
        self.pooling1 = dglnn.SetTransformerEncoder(time_L, n_heads=4, 
                                                    d_head=4, d_ff=256,
                                                    n_layers=2)
        self.pooling2 = dglnn.SetTransformerDecoder(time_L, num_heads=4, 
                                                    d_head=4, d_ff=256,
                                                    n_layers=2,
                                                    k=mode_N)       
        self.GNN = SAGE(fft_n+2, dim, dim, dropout_rate, hid_layer)
        self.node_decoder = MLP(dim, dim, mode_N, dropout_rate, hid_layer)
        self.time_L = time_L
        self.fft_n = fft_n
    def forward(self, g):
        node_in = g.ndata['acc_Y']
        # graph-level operation ##################################
        graph_spatial = self.pooling1(g, node_in)
        graph_spatial = self.pooling2(g, graph_spatial)
        [B, LK] = graph_spatial.size()
        q = graph_spatial.view(B, self.time_L, int(LK/self.time_L))    
        # node-level operation ##################################
        node_in_fft_abs = torch.fft.rfft(node_in, n=self.fft_n).abs()
        node_in_fft_angle = torch.fft.rfft(node_in, n=self.fft_n).angle()
        node_in_fft = torch.cat([node_in_fft_abs, node_in_fft_angle], dim=1)
        node_spatial = self.GNN(g, node_in_fft)
        phi = self.node_decoder(node_spatial)
        phi = phi / torch.max(torch.abs(phi), dim=0)[0]
        return q, phi # return mode responses and mode shapes

class Model_GIN(nn.Module):   # GIN
    def __init__(self, dim, time_L, fft_n, mode_N, dropout_rate, hid_layer):
        super().__init__()
        # use built-in Transformer ##################################
        self.pooling1 = dglnn.SetTransformerEncoder(time_L, n_heads=4, 
                                                    d_head=4, d_ff=256,
                                                    n_layers=2)
        self.pooling2 = dglnn.SetTransformerDecoder(time_L, num_heads=4, 
                                                    d_head=4, d_ff=256,
                                                    n_layers=2,
                                                    k=mode_N)       
        self.GNN = GIN(fft_n+2, dim)
        self.node_decoder = MLP(dim, dim, mode_N, dropout_rate, hid_layer)
        self.time_L = time_L
        self.fft_n = fft_n
    def forward(self, g):
        node_in = g.ndata['acc_Y']
        # graph-level operation ##################################
        graph_spatial = self.pooling1(g, node_in)
        graph_spatial = self.pooling2(g, graph_spatial)
        [B, LK] = graph_spatial.size()
        q = graph_spatial.view(B, self.time_L, int(LK/self.time_L))    
        # node-level operation ##################################
        node_in_fft_abs = torch.fft.rfft(node_in, n=self.fft_n).abs()
        node_in_fft_angle = torch.fft.rfft(node_in, n=self.fft_n).angle()
        node_in_fft = torch.cat([node_in_fft_abs, node_in_fft_angle], dim=1)
        node_spatial = self.GNN(g, node_in_fft)
        phi = self.node_decoder(node_spatial)
        phi = phi / torch.max(torch.abs(phi), dim=0)[0]
        return q, phi # return mode responses and mode shapes

def create_model(model_name, dim, time_L, fft_n, mode_N, dropout_rate, hid_layer):
    if model_name == 'SAGE':
        return Model_SAGE(dim, time_L, fft_n, mode_N, dropout_rate, hid_layer)
    elif model_name == 'GIN':
        return Model_GIN(dim, time_L, fft_n, mode_N, dropout_rate, hid_layer)
    else:
        raise ValueError(f"Model {model_name} not recognized.")