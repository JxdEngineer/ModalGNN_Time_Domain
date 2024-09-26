# pytorch lib
import torch
import torch.nn as nn
import dgl.nn as dglnn

# Define GNNs
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

# Define a CNN
class EncoderCNN(nn.Module):
    def __init__(self):
        super(EncoderCNN, self).__init__()
        # First Conv Layer: Reduce the 513 length to a smaller size
        # First Conv Layer: Input is (N, 1, 513, 2)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(5, 2), stride=(2, 1), padding=(2, 0))
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(5, 1), stride=(2, 1), padding=(2, 0))
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 1), stride=(2, 1), padding=(2, 0))
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 1), stride=(2, 1), padding=(2, 0))     
        # Global Average Pooling to get down to (N, 128, 64)
        self.global_pool = nn.AdaptiveAvgPool2d((64, 1))
        self.activation = torch.nn.Tanh()
        self.dropout = nn.Dropout(p=0.2)
        # Output linear layer
        self.fc = nn.Linear(128, 1) 
    def forward(self, x0):
        # Reshape input to (N, 1, 513, 2)
        x = x0.view(x0.size(0), 1, 513, 2)  
        x = self.conv1(x)  # (N, 16, 257, 1)
        x = self.dropout(self.activation(x))
        x = self.conv2(x)  # (N, 32, 129, 1)
        x = self.dropout(self.activation(x))
        x = self.conv3(x)  # (N, 64, 65, 1)
        x = self.dropout(self.activation(x))
        x = self.conv4(x)  # (N, 128, 33, 1)
        x = self.dropout(self.activation(x))
        x = self.global_pool(x)    # (N, 128, 64, 1)
        x = x.squeeze(-1)          # Remove last dimension to get (N, 128, 64)
        x = x.permute(0, 2, 1)     # Change to (N, 64, 128)
        x = self.fc(x)     # (N, 64, 1)
        x = x.squeeze()
        return x

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

# class Model_TEST(nn.Module):   # GraphSAGE, use two GNNs and MLPs
#     def __init__(self, dim, time_L, fft_n, mode_N, dropout_rate, hid_layer):
#         super().__init__()
#         # use built-in Transformer ##################################
#         self.pooling1 = dglnn.SetTransformerEncoder(time_L, n_heads=4, 
#                                                     d_head=4, d_ff=256,
#                                                     n_layers=2)
#         self.pooling2 = dglnn.SetTransformerDecoder(time_L, num_heads=4, 
#                                                     d_head=4, d_ff=256,
#                                                     n_layers=2,
#                                                     k=mode_N)       
#         self.GNN1 = SAGE(round(fft_n/2)+1, dim, dim, dropout_rate, hid_layer)
#         self.GNN2 = SAGE(round(fft_n/2)+1, dim, dim, dropout_rate, hid_layer)
#         self.node_decoder1 = MLP(dim, dim, mode_N, dropout_rate, hid_layer)
#         self.node_decoder2 = MLP(dim, dim, mode_N, dropout_rate, hid_layer)
#         self.time_L = time_L
#         self.fft_n = fft_n
#     def forward(self, g):
#         node_in = g.ndata['acc_Y']
#         # graph-level operation ##################################
#         graph_spatial = self.pooling1(g, node_in)
#         graph_spatial = self.pooling2(g, graph_spatial)
#         [B, LK] = graph_spatial.size()
#         q = graph_spatial.view(B, self.time_L, int(LK/self.time_L))    
#         # node-level operation ##################################
#         node_in_fft_abs = torch.fft.rfft(node_in, n=self.fft_n).abs()
#         node_in_fft_angle = torch.fft.rfft(node_in, n=self.fft_n).angle()
#         node_spatial1 = self.GNN1(g, node_in_fft_abs)
#         node_spatial2 = self.GNN2(g, node_in_fft_angle)
#         phi_abs = self.node_decoder1(node_spatial1)
#         phi_sign = torch.sign(self.node_decoder2(node_spatial2))
#         phi = phi_abs*phi_sign
#         phi = phi / torch.max(torch.abs(phi), dim=0)[0]
#         return q, phi # return mode responses and mode shapes

# class Model_TEST(nn.Module):   # GraphSAGE, use FFT real and imag as input
#     def __init__(self, dim, time_L, fft_n, mode_N, dropout_rate, hid_layer):
#         super().__init__()
#         # use built-in Transformer ##################################
#         self.pooling1 = dglnn.SetTransformerEncoder(time_L, n_heads=4, 
#                                                     d_head=4, d_ff=256,
#                                                     n_layers=2)
#         self.pooling2 = dglnn.SetTransformerDecoder(time_L, num_heads=4, 
#                                                     d_head=4, d_ff=256,
#                                                     n_layers=2,
#                                                     k=mode_N)       
#         self.GNN = SAGE(fft_n+2, dim, dim, dropout_rate, hid_layer)
#         self.node_decoder = MLP(dim, dim, mode_N, dropout_rate, hid_layer)
#         self.time_L = time_L
#         self.fft_n = fft_n
#     def forward(self, g):
#         node_in = g.ndata['acc_Y']
#         # graph-level operation ##################################
#         graph_spatial = self.pooling1(g, node_in)
#         graph_spatial = self.pooling2(g, graph_spatial)
#         [B, LK] = graph_spatial.size()
#         q = graph_spatial.view(B, self.time_L, int(LK/self.time_L))    
#         # node-level operation ##################################
#         node_in_fft_real = torch.fft.rfft(node_in, n=self.fft_n).real
#         node_in_fft_imag = torch.fft.rfft(node_in, n=self.fft_n).imag
#         node_in_fft = torch.cat([node_in_fft_real, node_in_fft_imag], dim=1)
#         node_spatial = self.GNN(g, node_in_fft)
#         phi = self.node_decoder(node_spatial)
#         phi = phi / torch.max(torch.abs(phi), dim=0)[0]
#         return q, phi # return mode responses and mode shapes

class Model_TEST(nn.Module):   # GraphSAGE, use CNN to encode the FFT
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
        self.GNN = SAGE(dim, dim, dim, dropout_rate, hid_layer)
        self.node_encoder = EncoderCNN()
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
        node_in_fft = torch.stack([node_in_fft_abs, node_in_fft_angle], dim=2)
        node_in_hid = self.node_encoder(node_in_fft)
        node_spatial = self.GNN(g, node_in_hid)
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
        node_in_fft = torch.cat([node_in_fft_abs, node_in_fft_angle], dim=0)
        node_spatial = self.GNN(g, node_in_fft)
        phi = self.node_decoder(node_spatial)
        phi = phi / torch.max(torch.abs(phi), dim=0)[0]
        return q, phi # return mode responses and mode shapes

def create_model(model_name, dim, time_L, fft_n, mode_N, dropout_rate, hid_layer):
    if model_name == 'SAGE':
        return Model_SAGE(dim, time_L, fft_n, mode_N, dropout_rate, hid_layer)
    elif model_name == 'GIN':
        return Model_GIN(dim, time_L, fft_n, mode_N, dropout_rate, hid_layer)
    elif model_name == 'TEST':
        return Model_TEST(dim, time_L, fft_n, mode_N, dropout_rate, hid_layer)
    else:
        raise ValueError(f"Model {model_name} not recognized.")