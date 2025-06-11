# pytorch lib
import torch
import torch.nn as nn
import dgl
import dgl.nn as dglnn

# Define GNNs
class SAGE(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, dropout_rate, hid_layer):
        super().__init__()
        self.conv1 = dglnn.SAGEConv(in_feats, hid_feats, aggregator_type='pool',
                                    feat_drop=dropout_rate, activation=nn.Tanh(),
                                    bias=True, norm=nn.BatchNorm1d(hid_feats))
        self.conv2 = dglnn.SAGEConv(hid_feats, hid_feats, aggregator_type='pool',
                                    feat_drop=dropout_rate, activation=nn.Tanh(),
                                    bias=True, norm=nn.BatchNorm1d(hid_feats))
        self.conv3 = dglnn.SAGEConv(hid_feats, out_feats, aggregator_type='pool',
                                    feat_drop=0, activation=None,
                                    bias=True)
        self.N_hid_layer = hid_layer
    def forward(self, graph, inputs):
        h = self.conv1(graph, inputs)
        for i in range(self.N_hid_layer):
            h = self.conv2(graph, h)
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
        self.N_hid_layer = hid_layer
    def forward(self, x):
        y = self.layer1(x)
        y = self.activation(y)
        y = self.dropout(y)
        for i in range(self.N_hid_layer):
            y = self.layer2(y)
            y = self.activation(y)
            y = self.dropout(y)
        y = self.layer3(y)
        return y

# Define a Transformer
class Transformer(nn.Module):   # ablation study, no GNN model
    def __init__(self, dim, mode_N, dropout_rate):
        super().__init__()
        # Input Projection (Map 5 Channels → embed_dim)
        self.input_projection = nn.Linear(7, dim) 
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=4, 
                                                   dim_feedforward=256, 
                                                   dropout=dropout_rate, 
                                                   batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        # Output Projection (Map embed_dim → mode_N Channels)
        self.output_projection = nn.Linear(dim, mode_N)
    def forward(self, x):
        x = self.input_projection(x)
        x = self.encoder(x)
        x = self.output_projection(x)
        return x

# Define the entire model
class Model_Benchmark(nn.Module):   # GraphSAGE, new architecture, GNN first, Transformer readout next
    def __init__(self, dim, time_L, mode_N, dropout_rate, hid_layer):
        super().__init__()
        self.pooling1 = dglnn.SetTransformerEncoder(time_L, n_heads=4, 
                                                    d_head=4, d_ff=256,
                                                    n_layers=1, 
                                                    dropouth=dropout_rate)
        self.pooling2 = dglnn.SetTransformerDecoder(time_L, num_heads=4, 
                                                    d_head=4, d_ff=256,
                                                    n_layers=1,
                                                    k=dim, 
                                                    dropouth=dropout_rate)
        self.graph_decoder = MLP(dim, dim, mode_N, dropout_rate, hid_layer)
        
        self.GNN = SAGE(time_L, dim, time_L, dropout_rate, hid_layer)
        
        self.node_decoder = MLP(time_L, dim, mode_N, dropout_rate, hid_layer)
        
        self.time_L = time_L
    def forward(self, g):
        node_in = g.ndata['acc_Y']
        node_spatial = self.GNN(g, node_in)
        # graph-level operation ##################################
        graph_spatial = self.pooling1(g, node_spatial)
        graph_spatial = self.pooling2(g, graph_spatial)
        [B, LK] = graph_spatial.size()
        q = graph_spatial.view(B, self.time_L, int(LK/self.time_L))
        q = self.graph_decoder(q)
        # node-level operation ##################################
        phi = self.node_decoder(node_spatial)
        
        # normalization
        g_unbatched = dgl.unbatch(g)
        graph_sizes = [graph.num_nodes() for graph in g_unbatched]
        split_phi = torch.split(phi, graph_sizes)  # Split `phi` based on graph sizes
        phi_norm_list = [
            phi_graph / torch.max(torch.abs(phi_graph), dim=0, keepdim=True)[0] 
            for phi_graph in split_phi
        ]
            
        phi_norm = torch.cat(phi_norm_list, dim=0)
        
        return q, phi_norm # return mode responses and mode shapes


class Model_NoGNN(nn.Module):   # ablation study, no GNN model
    def __init__(self, dim, time_L, mode_N, dropout_rate, hid_layer):
        super().__init__()
        self.q_identifier = Transformer(dim, mode_N, dropout_rate)
        
        self.phi_identifier = MLP(time_L, dim, mode_N, dropout_rate, hid_layer)
        
        self.time_L = time_L
    def forward(self, g):
        # identify modal response ##################################
        g_unbatched = dgl.unbatch(g)
        q_list = []  # List to store q tensors
        for i in range(len(g_unbatched)):
            node_in_unbatched = g_unbatched[i].ndata['acc_Y']
            mask = g_unbatched[i].ndata['mask']
            node_in_unbatched_mask = node_in_unbatched[mask, :]
            q_unbatched = self.q_identifier(node_in_unbatched_mask[2:9, :].T)
            q_list.append(q_unbatched)  # Store in list
        q = torch.stack(q_list, dim=0)  # Shape: (bs, 2000, 7)
        # identify mode shapes ##################################
        node_in = g.ndata['acc_Y']
        phi = self.phi_identifier(node_in)
        
        # normalization
        graph_sizes = [graph.num_nodes() for graph in g_unbatched]
        split_phi = torch.split(phi, graph_sizes)  # Split `phi` based on graph sizes
        phi_norm_list = [
            phi_graph / torch.max(torch.abs(phi_graph), dim=0, keepdim=True)[0] 
            for phi_graph in split_phi
        ]
            
        phi_norm = torch.cat(phi_norm_list, dim=0)
        
        return q, phi_norm # return mode responses and mode shapes

class Model_LSTM(nn.Module):   # GraphSAGE, new architecture, GNN first, LSTM readout next
    def __init__(self, dim, time_L, mode_N, dropout_rate, hid_layer):
        super().__init__()
        self.pooling = dglnn.Set2Set(time_L, n_iters=2, n_layers=2)
        self.graph_decoder = MLP(2, dim, mode_N, dropout_rate, hid_layer)
        self.GNN = SAGE(time_L, dim, time_L, dropout_rate, hid_layer)
        self.node_decoder = MLP(time_L, dim, mode_N, dropout_rate, hid_layer)
        self.time_L = time_L
    def forward(self, g):
        node_in = g.ndata['acc_Y']
        node_spatial = self.GNN(g, node_in)
        # graph-level operation ##################################
        graph_spatial = self.pooling(g, node_spatial)

        [B, LK] = graph_spatial.size()
        q = graph_spatial.view(B, self.time_L, int(LK/self.time_L))
        q = self.graph_decoder(q)
        # node-level operation ##################################
        phi = self.node_decoder(node_spatial)
        
        # normalization
        g_unbatched = dgl.unbatch(g)
        graph_sizes = [graph.num_nodes() for graph in g_unbatched]
        split_phi = torch.split(phi, graph_sizes)  # Split `phi` based on graph sizes
        phi_norm_list = [
            phi_graph / torch.max(torch.abs(phi_graph), dim=0, keepdim=True)[0] 
            for phi_graph in split_phi
        ]
            
        phi_norm = torch.cat(phi_norm_list, dim=0)
        
        return q, phi_norm # return mode responses and mode shapes

def create_model(model_name, dim, time_L, mode_N, dropout_rate, hid_layer):
    if model_name == 'Benchmark':
        return Model_Benchmark(dim, time_L, mode_N, dropout_rate, hid_layer)
    elif model_name == 'NoGNN':
        return Model_NoGNN(dim, time_L, mode_N, dropout_rate, hid_layer)
    elif model_name == 'LSTM':
        return Model_LSTM(dim, time_L, mode_N, dropout_rate, hid_layer)
    else:
        raise ValueError(f"Model {model_name} not recognized.")