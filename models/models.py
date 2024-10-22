# pytorch lib
import torch
import torch.nn as nn
import dgl
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

class SAGE_Residual(nn.Module):
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
        hr = h
        # h = self.norm(h)
        h = self.activation(h + hr)
        h = self.dropout(h)
        for i in range(self.hid_layer):
            h = self.conv2(graph, h)
            hr = h
            # h = self.norm(h)
            h = self.activation(h + hr)
            h = self.dropout(h)
        h = self.conv3(graph, h)
        return h

class GIN(nn.Module):
    def __init__(self, in_feats, out_feats, dropout_rate):
        super().__init__()
        self.conv1 = dglnn.GINConv(nn.Linear(in_feats, out_feats))
        self.activation = torch.nn.Tanh()
        self.dropout = nn.Dropout(p=dropout_rate)
    def forward(self, graph, node_inputs):
        h = self.conv1(graph, node_inputs)
        h = self.activation(h)
        h = self.dropout(h)
        return h

class GINE(nn.Module):
    def __init__(self, dropout_rate):
        super().__init__()
        self.conv = dglnn.GINEConv()
        self.activation = torch.nn.Tanh()
        self.dropout = nn.Dropout(p=dropout_rate)
    def forward(self, graph, node_inputs, edge_inputs):
        h = self.conv(graph, node_inputs, edge_inputs)
        h = self.activation(h)
        h = self.dropout(h)
        h = self.conv(graph, node_inputs, edge_inputs)
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
        h = self.dropout(h)
        for i in range(self.hid_layer):
            h = self.conv2(graph, h)
            h = self.activation(h)
            h = self.dropout(h)
        h = self.conv3(graph, h)
        return h

class EdgeGAT(nn.Module):
    def __init__(self, node_feats, hid_feats, out_feats, edge_feats, dropout_rate, hid_layer):
        super().__init__()
        self.conv1 = dglnn.EdgeGATConv(node_feats, edge_feats, hid_feats, num_heads=1)
        self.conv2 = dglnn.EdgeGATConv(hid_feats, edge_feats, hid_feats, num_heads=1)
        self.conv3 = dglnn.EdgeGATConv(hid_feats, edge_feats, out_feats, num_heads=1)
        self.activation = torch.nn.Tanh()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.hid_layer = hid_layer
    def forward(self, graph, node_in, edge_in):
        h = self.conv1(graph, node_in, edge_in).squeeze()
        h = self.activation(h)
        h = self.dropout(h)
        for i in range(self.hid_layer):
            h = self.conv2(graph, h, edge_in).squeeze()
            h = self.activation(h)
            h = self.dropout(h)
        h = self.conv3(graph, h, edge_in).squeeze()
        return h

class GatedGraphConv(nn.Module):
    def __init__(self, node_in_feats, out_feats, dropout_rate):
        super().__init__()
        self.conv1 = dglnn.GatedGraphConv(node_in_feats, out_feats, n_steps=3, n_etypes=1)
        self.activation = torch.nn.Tanh()
        self.dropout = nn.Dropout(p=dropout_rate)
    def forward(self, graph, node_in):
        h = self.conv1(graph, node_in)
        h = self.activation(h)
        h = self.dropout(h)
        h = self.conv1(graph, node_in)
        return h

class CFC(nn.Module):
    def __init__(self, node_in_feats, edge_in_feats, hid_feats, out_feats, dropout_rate):
        super().__init__()
        self.conv1 = dglnn.CFConv(node_in_feats, edge_in_feats, hid_feats, hid_feats)
        self.conv2 = dglnn.CFConv(hid_feats, edge_in_feats, hid_feats, out_feats)
        self.activation = torch.nn.Tanh()
        self.dropout = nn.Dropout(p=dropout_rate)
    def forward(self, graph, node_inputs, edge_inputs):
        h = self.conv1(graph, node_inputs, edge_inputs)
        h = self.activation(h)
        h = self.dropout(h)
        h = self.conv2(graph, h, edge_inputs)
        h = self.activation(h)
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

# Define a Transformer
class TransformerTimeSeries(nn.Module):
    def __init__(self, d_model=64, nhead=4, num_layers=4, seq_length=2000):
        super().__init__()
        # Linear layer to embed N-D time series to d_model dimensions
        self.embedding = nn.Linear(3, d_model) 
        # Positional encoding to give the transformer a sense of order
        self.positional_encoding = nn.Parameter(torch.zeros(3, seq_length, d_model))
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # Final linear layer to convert d_model output to 7 dimensions
        self.fc = nn.Linear(d_model, 7)
    def forward(self, x):
        # x is of shape (batch_size, seq_length, 1)
        # Embed the 1D time series into a higher-dimensional space
        x1 = self.embedding(x)  # (batch_size, seq_length, d_model)
        # Add positional encoding
        # x1 = x1 + nn.Parameter(torch.zeros(x1.size()))
        # Pass through the transformer encoder
        x1 = self.transformer(x1)  # (batch_size, seq_length, d_model)
        # Project the transformer output to 7 dimensions
        x1 = self.fc(x1)  # (batch_size, seq_length, 7)
        return x1
# Define a CNN
# class CNN1D(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # First 1D convolutional layer: input 2 channels (magnitude + phase) -> 32 filters
#         self.conv1 = nn.Conv1d(in_channels=2, out_channels=32, kernel_size=5, stride=1, padding=2)
#         # Second 1D convolutional layer: 32 filters -> 64 filters
#         self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2) 
#         # Pooling layer to reduce dimensionality along the frequency axis
#         self.pool = nn.MaxPool1d(kernel_size=2, stride=2) 
#         # Third 1D convolutional layer: 64 filters -> 128 filters
#         self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1) 
#         # Fully connected layer to transform to N*64 features
#         self.fc = nn.Linear(128*64, 64)  # Adjust 128*128 based on pooling/convolution output size
#         self.activation = torch.nn.Tanh()
#         self.dropout = nn.Dropout(p=0.2)
#     def forward(self, x):
#         x = x.permute(0, 2, 1)
#         # Apply conv1 -> activation -> pooling
#         x = self.pool(self.dropout(self.activation(self.conv1(x))))  # Shape: [N, 32, 256]
#         # Apply conv2 -> activation -> pooling
#         x = self.pool(self.dropout(self.activation(self.conv2(x))))  # Shape: [N, 64, 128]
#         # Apply conv3 -> activation -> pooling
#         x = self.pool(self.dropout(self.activation(self.conv3(x))))  # Shape: [N, 128, 64]
#         # Flatten the output for the fully connected layer
#         x = x.view(x.size(0), -1)  # Flatten to [N, 128*64]
#         # Fully connected layer to produce N*64 output
#         x = self.fc(x)  # Shape: [N, 64]
#         return x

class CNN1D(nn.Module):
    def __init__(self):
        super().__init__()
        # self.conv1d = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=8, stride=8)
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=9, stride=8, padding=1)
        # self.conv2 = nn.Conv1d(in_channels=16, out_channels=1, kernel_size=9, stride=1, padding=4)
        self.activation = torch.nn.Tanh()
        self.dropout = nn.Dropout(p=0.2)
    def forward(self, x):
        # Input: N*2*513 -> Output: N*1*64
        x1 = self.conv1(x)
        x1 = self.activation(x1)
        x1 = self.dropout(x1)
        # x1 = self.conv2(x1)
        # N*1*64 -> N*64
        x1 = x1.squeeze()
        return x1

class CNN2D(nn.Module):
    def __init__(self):
        super().__init__()
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
# class Model_SAGE(nn.Module):   # GraphSAGE, use cat(FFT.abs, FFT.angle) as node input -------------- benchmark
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
#         # self.norm = nn.BatchNorm1d(round((fft_n+2)/2))
#         # self.norm = nn.BatchNorm1d(time_L)
#     def forward(self, g):
#         node_in = g.ndata['acc_Y']
#         # node_in = self.norm(node_in)
#         # graph-level operation ##################################
#         graph_spatial = self.pooling1(g, node_in)
#         graph_spatial = self.pooling2(g, graph_spatial)
#         [B, LK] = graph_spatial.size()
#         q = graph_spatial.view(B, self.time_L, int(LK/self.time_L))    
#         # node-level operation ##################################
#         node_in_fft_abs = torch.fft.rfft(node_in, n=self.fft_n).abs()
#         node_in_fft_angle = torch.fft.rfft(node_in, n=self.fft_n).angle() / 1
#         # normalization of features
#         # node_in_fft_abs = self.norm(node_in_fft_abs)
#         # node_in_fft_angle = self.norm(node_in_fft_angle)
#         node_in_fft = torch.cat([node_in_fft_abs, node_in_fft_angle], dim=1)       
#         node_spatial = self.GNN(g, node_in_fft)
#         phi = self.node_decoder(node_spatial)
#         phi = phi / torch.max(torch.abs(phi), dim=0)[0]
#         return q, phi # return mode responses and mode shapes

# class Model_SAGE(nn.Module):   # GraphSAGE, use cat(FFT.abs, FFT.angle_smoothed) as node input
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
#         # self.norm = nn.BatchNorm1d(round((fft_n+2)/2))
#         # self.norm = nn.BatchNorm1d(time_L)
#     def forward(self, g):
#         node_in = g.ndata['acc_Y']
#         # node_in = self.norm(node_in)
#         # graph-level operation ##################################
#         graph_spatial = self.pooling1(g, node_in)
#         graph_spatial = self.pooling2(g, graph_spatial)
#         [B, LK] = graph_spatial.size()
#         q = graph_spatial.view(B, self.time_L, int(LK/self.time_L))    
#         # node-level operation ##################################
#         node_in_fft_abs = g.ndata['acc_Y_FFT_abs'] * 10
#         node_in_fft_angle = g.ndata['acc_Y_FFT_angle'] / 3.14
#         node_in_fft = torch.cat([node_in_fft_abs, node_in_fft_angle], dim=1)       
#         node_spatial = self.GNN(g, node_in_fft)
#         phi = self.node_decoder(node_spatial)
#         phi = phi / torch.max(torch.abs(phi), dim=0)[0]
#         return q, phi # return mode responses and mode shapes

# class Model_SAGE(nn.Module):   # GraphSAGE, use FFT.abs*FFT.angle as node input
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
#         self.GNN = SAGE(round((fft_n+2)/2), dim, dim, dropout_rate, hid_layer)
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
#         node_in_fft_abs = torch.fft.rfft(node_in, n=self.fft_n).abs()
#         node_in_fft_angle = torch.fft.rfft(node_in, n=self.fft_n).angle()
#         node_in_fft = node_in_fft_abs * node_in_fft_angle  
#         node_spatial = self.GNN(g, node_in_fft)
#         phi = self.node_decoder(node_spatial)
#         phi = phi / torch.max(torch.abs(phi), dim=0)[0]
#         return q, phi # return mode responses and mode shapes

# class Model_SAGE(nn.Module):   # GraphSAGE, new architecture, GNN first, Transformer readout next
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
#         self.GNN = SAGE(time_L, dim, time_L, dropout_rate, hid_layer)
#         self.node_decoder = MLP(time_L, dim, mode_N, dropout_rate, hid_layer)
#         self.time_L = time_L
#         self.fft_n = fft_n
#     def forward(self, g):
#         node_in = g.ndata['acc_Y']
#         node_spatial = self.GNN(g, node_in)
#         # graph-level operation ##################################
#         graph_spatial = self.pooling1(g, node_spatial)
#         graph_spatial = self.pooling2(g, graph_spatial)
#         [B, LK] = graph_spatial.size()
#         q = graph_spatial.view(B, self.time_L, int(LK/self.time_L))    
#         # node-level operation ##################################
#         phi = self.node_decoder(node_spatial)
#         phi = phi / torch.max(torch.abs(phi), dim=0)[0]
#         return q, phi # return mode responses and mode shapes

# class Model_SAGE(nn.Module):   # GraphSAGE, new architecture, GNN first, max/min/mean/WeightAndSum readout next, then use LSTM
#     def __init__(self, dim, time_L, fft_n, mode_N, dropout_rate, hid_layer):
#         super().__init__()
#         # use built-in Transformer ##################################
#         self.GNN = SAGE(time_L, dim, time_L, dropout_rate, hid_layer)
#         self.pooling1 = dglnn.WeightAndSum(time_L)
#         self.pooling2 = dglnn.MaxPooling()
#         self.pooling3 = dglnn.AvgPooling()
#         self.pooling4 = dglnn.SumPooling()
#         self.graph_decoder = nn.LSTM(input_size=4, hidden_size=64, num_layers=2, batch_first=True)
#         self.LSTM_decoder = MLP(64, dim, mode_N, dropout_rate, hid_layer)
#         self.node_decoder = MLP(time_L, dim, mode_N, dropout_rate, hid_layer)
#         self.time_L = time_L
#         self.fft_n = fft_n
#     def forward(self, g):
#         node_in = g.ndata['acc_Y']
#         node_spatial = self.GNN(g, node_in)
#         # graph-level operation ##################################
#         graph_spatial = torch.stack((self.pooling1(g, node_spatial),
#                                     self.pooling2(g, node_spatial),
#                                     self.pooling3(g, node_spatial),
#                                     self.pooling4(g, node_spatial)), dim=2)
#         q, _ = self.graph_decoder(graph_spatial)
#         q = self.LSTM_decoder(q)
#         # node-level operation ##################################
#         phi = self.node_decoder(node_spatial)
#         phi = phi / torch.max(torch.abs(phi), dim=0)[0]
#         return q, phi # return mode responses and mode shapes

# class Model_SAGE(nn.Module):   # GraphSAGE, new architecture, GNN first, max/min/mean readout next, then use LSTM+MLP to decode
#     def __init__(self, dim, time_L, fft_n, mode_N, dropout_rate, hid_layer):
#         super().__init__()
#         # use built-in Transformer ##################################
#         self.GNN = SAGE(time_L, dim, time_L, dropout_rate, hid_layer)
#         self.pooling2 = dglnn.MaxPooling()
#         self.pooling3 = dglnn.AvgPooling()
#         self.pooling4 = dglnn.SumPooling()
#         self.graph_decoder = nn.LSTM(input_size=3, hidden_size=64, num_layers=2, batch_first=True)
#         self.LSTM_decoder = MLP(64, dim, mode_N, dropout_rate, hid_layer)
#         self.node_decoder = MLP(time_L, dim, mode_N, dropout_rate, hid_layer)
#         self.time_L = time_L
#         self.fft_n = fft_n
#     def forward(self, g):
#         node_in = g.ndata['acc_Y']
#         node_spatial = self.GNN(g, node_in)
#         # graph-level operation ##################################
#         graph_spatial = torch.stack((self.pooling2(g, node_spatial),
#                                     self.pooling3(g, node_spatial),
#                                     self.pooling4(g, node_spatial)), dim=2)
#         q, _ = self.graph_decoder(graph_spatial)
#         q = self.LSTM_decoder(q)
#         # node-level operation ##################################
#         phi = self.node_decoder(node_spatial)
#         phi = phi / torch.max(torch.abs(phi), dim=0)[0]
#         return q, phi # return mode responses and mode shapes

# class Model_SAGE(nn.Module):   # GraphSAGE, new architecture, GNN first, max/min/mean readout next, then use Transformer to decode
#     def __init__(self, dim, time_L, fft_n, mode_N, dropout_rate, hid_layer):
#         super().__init__()
#         # use built-in Transformer ##################################
#         self.GNN = SAGE(time_L, dim, time_L, dropout_rate, hid_layer)
#         self.pooling2 = dglnn.MaxPooling()
#         self.pooling3 = dglnn.AvgPooling()
#         self.pooling4 = dglnn.SumPooling()
#         self.graph_decoder = TransformerTimeSeries(d_model=64, nhead=4, num_layers=4, seq_length=2000)
#         self.node_decoder = MLP(time_L, dim, mode_N, dropout_rate, hid_layer)
#         self.time_L = time_L
#         self.fft_n = fft_n
#     def forward(self, g):
#         node_in = g.ndata['acc_Y']
#         node_spatial = self.GNN(g, node_in)
#         # graph-level operation ##################################
#         graph_spatial = torch.stack((self.pooling2(g, node_spatial),
#                                     self.pooling3(g, node_spatial),
#                                     self.pooling4(g, node_spatial)), dim=2)
#         q = self.graph_decoder(graph_spatial)
#         # node-level operation ##################################
#         phi = self.node_decoder(node_spatial)
#         phi = phi / torch.max(torch.abs(phi), dim=0)[0]
#         return q, phi # return mode responses and mode shapes

# class Model_SAGE(nn.Module):   # GraphSAGE, new architecture, GNN first, max/min/mean/WeightAndSum readout next, then use MLP to decode
#     def __init__(self, dim, time_L, fft_n, mode_N, dropout_rate, hid_layer):
#         super().__init__()
#         # use built-in Transformer ##################################
#         self.GNN = SAGE(time_L, dim, time_L, dropout_rate, hid_layer)
#         self.pooling1 = dglnn.WeightAndSum(time_L)
#         self.pooling2 = dglnn.MaxPooling()
#         self.pooling3 = dglnn.AvgPooling()
#         self.pooling4 = dglnn.SumPooling()
#         self.graph_decoder = MLP(4, dim, mode_N, dropout_rate, hid_layer)
#         self.node_decoder = MLP(time_L, dim, mode_N, dropout_rate, hid_layer)
#         self.time_L = time_L
#         self.fft_n = fft_n
#     def forward(self, g):
#         node_in = g.ndata['acc_Y']
#         node_spatial = self.GNN(g, node_in)
#         # graph-level operation ##################################
#         graph_spatial = torch.stack((self.pooling1(g, node_spatial),
#                                     self.pooling2(g, node_spatial),
#                                     self.pooling3(g, node_spatial),
#                                     self.pooling4(g, node_spatial)), dim=2)
#         q = self.graph_decoder(graph_spatial)
#         # node-level operation ##################################
#         phi = self.node_decoder(node_spatial)
#         phi = phi / torch.max(torch.abs(phi), dim=0)[0]
#         return q, phi # return mode responses and mode shapes

class Model_SAGE(nn.Module):   # GraphSAGE, GNN first, choose the most important 10 nodes, then use LSTM+MLP to get modal responses q
    def __init__(self, dim, time_L, fft_n, mode_N, dropout_rate, hid_layer):
        super().__init__()
        # use built-in Transformer ##################################
        self.GNN = SAGE(time_L, dim, time_L, dropout_rate, hid_layer)
        self.score = dglnn.SAGEConv(time_L, 1, aggregator_type='pool')
        self.graph_decoder = nn.LSTM(input_size=20, hidden_size=64, num_layers=4, batch_first=True)
        self.LSTM_decoder = MLP(64, dim, mode_N, dropout_rate, hid_layer)
        self.node_decoder = MLP(time_L, dim, mode_N, dropout_rate, hid_layer)
        self.time_L = time_L
        self.fft_n = fft_n
    def forward(self, g):
        node_in = g.ndata['acc_Y']
        node_spatial = self.GNN(g, node_in)
        g.ndata['node_spatial'] = node_spatial
        # graph-level operation ##################################
        graph_spatial = []
        for graphs in dgl.unbatch(g):
            scores = self.score(graphs, graphs.ndata['node_spatial']).squeeze()  # Shape: (num_nodes,)
            _, topk = torch.topk(scores, 20, dim=0)
            graph_spatial.append(graphs.ndata['node_spatial'][topk].T)
        graph_spatial = torch.stack(graph_spatial, dim=0)
        q, _ = self.graph_decoder(graph_spatial)
        q = self.LSTM_decoder(q)
        # node-level operation ##################################
        phi = self.node_decoder(node_spatial)
        phi = phi / torch.max(torch.abs(phi), dim=0)[0]
        return q, phi # return mode responses and mode shapes

class Model_2SAGE_2MLP(nn.Module):   # GraphSAGE, use two GNNs and MLPs
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
        self.GNN1 = SAGE(round(fft_n/2)+1, dim, dim, dropout_rate, hid_layer)
        self.GNN2 = SAGE(round(fft_n/2)+1, dim, dim, dropout_rate, hid_layer)
        self.node_decoder1 = MLP(dim, dim, mode_N, dropout_rate, hid_layer)
        self.node_decoder2 = MLP(dim, dim, mode_N, dropout_rate, hid_layer)
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
        node_spatial1 = self.GNN1(g, node_in_fft_abs)
        node_spatial2 = self.GNN2(g, node_in_fft_angle*node_in_fft_abs) 
        phi_abs = self.node_decoder1(node_spatial1)
        phi_sign = torch.sign(self.node_decoder2(node_spatial2))
        phi = phi_abs*phi_sign
        phi = phi / torch.max(torch.abs(phi), dim=0)[0]
        return q, phi # return mode responses and mode shapes

class Model_2SAGE_1MLP(nn.Module):   # GraphSAGE, use two GNNs and ONE MLP
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
        self.GNN1 = SAGE(round(fft_n/2)+1, dim, dim, dropout_rate, hid_layer)
        self.GNN2 = SAGE(round(fft_n/2)+1, dim, dim, dropout_rate, hid_layer)
        self.node_decoder = MLP(dim*2, dim, mode_N, dropout_rate, hid_layer)
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
        node_spatial1 = self.GNN1(g, node_in_fft_abs)
        node_spatial2 = self.GNN2(g, node_in_fft_angle*node_in_fft_abs)
        node_spatial = torch.cat([node_spatial1, node_spatial2], dim=-1)
        phi = self.node_decoder(node_spatial)
        phi = phi / torch.max(torch.abs(phi), dim=0)[0]
        return q, phi # return mode responses and mode shapes

class Model_CNN1D(nn.Module):   # GraphSAGE, use CNN to encode the FFT
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
        self.node_encoder = CNN1D()
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
        node_in_fft = torch.stack([node_in_fft_abs, node_in_fft_angle], dim=1)
        node_in_hid = self.node_encoder(node_in_fft)
        node_spatial = self.GNN(g, node_in_hid)
        phi = self.node_decoder(node_spatial)
        phi = phi / torch.max(torch.abs(phi), dim=0)[0]
        return q, phi # return mode responses and mode shapes

# class Model_CNN1D(nn.Module):   # GraphSAGE, use CNN to encode the FFT, use smooth FFT
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
#         self.GNN = SAGE(dim, dim, dim, dropout_rate, hid_layer)
#         self.node_encoder = CNN1D()
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
#         node_in_fft_abs = g.ndata['acc_Y_FFT_abs'] * 10
#         node_in_fft_angle = g.ndata['acc_Y_FFT_angle'] / 3.14
#         node_in_fft = torch.stack([node_in_fft_abs, node_in_fft_angle], dim=1)
#         node_in_hid = self.node_encoder(node_in_fft)
#         node_spatial = self.GNN(g, node_in_hid)
#         phi = self.node_decoder(node_spatial)
#         phi = phi / torch.max(torch.abs(phi), dim=0)[0]
#         return q, phi # return mode responses and mode shapes

class Model_CNN2D(nn.Module):   # GraphSAGE, use CNN to encode the FFT
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
        self.node_encoder = CNN2D()
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

class Model_GINE(nn.Module):   # GINE, using geometry as edge features
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
        self.GNN = GINE(dropout_rate)
        self.edge_encoder = MLP(2, dim, fft_n+2, dropout_rate, hid_layer)
        self.node_decoder = MLP(fft_n+2, dim, mode_N, dropout_rate, hid_layer)
        self.time_L = time_L
        self.fft_n = fft_n
    def forward(self, g):
        node_in = g.ndata['acc_Y']
        edge_in = torch.cat([g.edata['L'], g.edata['theta']/360], dim=1)
        # graph-level operation ##################################
        graph_spatial = self.pooling1(g, node_in)
        graph_spatial = self.pooling2(g, graph_spatial)
        [B, LK] = graph_spatial.size()
        q = graph_spatial.view(B, self.time_L, int(LK/self.time_L))    
        # node-level operation ##################################
        node_in_fft_abs = torch.fft.rfft(node_in, n=self.fft_n ).abs()
        node_in_fft_angle = torch.fft.rfft(node_in, n=self.fft_n ).angle()
        node_in_fft = torch.cat([node_in_fft_abs, node_in_fft_angle], dim=1)
        edge_in_encoded = self.edge_encoder(edge_in)
        node_spatial = self.GNN(g, node_in_fft, edge_in_encoded)
        phi = self.node_decoder(node_spatial)
        phi = phi / torch.max(torch.abs(phi), dim=0)[0]
        return q, phi # return mode responses and mode shapes

# class Model_GINE(nn.Module):   # GINE, using CPSD.real as edge features
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
#         self.GNN = GINE(dropout_rate)
#         self.edge_encoder = MLP(round(fft_n/2)+1, dim, fft_n+2, dropout_rate, hid_layer)
#         self.node_decoder = MLP(fft_n+2, dim, mode_N, dropout_rate, hid_layer)
#         self.time_L = time_L
#         self.fft_n = fft_n
#     def forward(self, g):
#         node_in = g.ndata['acc_Y']
#         # edge_in = g.edata['cpsd_angle'] / 3.14
#         edge_in = g.edata['cpsd_real'] * 1e7
#         # graph-level operation ##################################
#         graph_spatial = self.pooling1(g, node_in)
#         graph_spatial = self.pooling2(g, graph_spatial)
#         [B, LK] = graph_spatial.size()
#         q = graph_spatial.view(B, self.time_L, int(LK/self.time_L))    
#         # node-level operation ##################################
#         node_in_fft_abs = torch.fft.rfft(node_in, n=self.fft_n ).abs()
#         node_in_fft_angle = torch.fft.rfft(node_in, n=self.fft_n ).angle()
#         node_in_fft = torch.cat([node_in_fft_abs, node_in_fft_angle], dim=1)
#         edge_in_encoded = self.edge_encoder(edge_in)
#         node_spatial = self.GNN(g, node_in_fft, edge_in_encoded)
#         phi = self.node_decoder(node_spatial)
#         phi = phi / torch.max(torch.abs(phi), dim=0)[0]
#         return q, phi # return mode responses and mode shapes

# class Model_EdgeGAT(nn.Module):   # GAT, using geometry as edge features
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
#         self.GNN = EdgeGAT(fft_n+2, dim, dim, 2, dropout_rate, hid_layer)
#         self.node_decoder = MLP(dim, dim, mode_N, dropout_rate, hid_layer)
#         self.time_L = time_L
#         self.fft_n = fft_n
#     def forward(self, g):
#         node_in = g.ndata['acc_Y']
#         edge_in = torch.cat([g.edata['L'], g.edata['theta']/360], dim=1)
#         # graph-level operation ##################################
#         graph_spatial = self.pooling1(g, node_in)
#         graph_spatial = self.pooling2(g, graph_spatial)
#         [B, LK] = graph_spatial.size()
#         q = graph_spatial.view(B, self.time_L, int(LK/self.time_L))    
#         # node-level operation ##################################
#         node_in_fft_abs = torch.fft.rfft(node_in, n=self.fft_n ).abs()
#         node_in_fft_angle = torch.fft.rfft(node_in, n=self.fft_n ).angle()
#         node_in_fft = torch.cat([node_in_fft_abs, node_in_fft_angle], dim=1)
#         node_spatial = self.GNN(g, node_in_fft, edge_in)
#         phi = self.node_decoder(node_spatial)
#         phi = phi / torch.max(torch.abs(phi), dim=0)[0]
#         return q, phi # return mode responses and mode shapes

class Model_EdgeGAT(nn.Module):   # GAT, using CPSD.real as edge features
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
        self.GNN = EdgeGAT(fft_n+2, dim, dim, round(fft_n/2)+1, dropout_rate, hid_layer)
        self.node_decoder = MLP(dim, dim, mode_N, dropout_rate, hid_layer)
        self.time_L = time_L
        self.fft_n = fft_n
    def forward(self, g):
        node_in = g.ndata['acc_Y']
        edge_in = g.edata['cpsd_real'] * 1e7
        # graph-level operation ##################################
        graph_spatial = self.pooling1(g, node_in)
        graph_spatial = self.pooling2(g, graph_spatial)
        [B, LK] = graph_spatial.size()
        q = graph_spatial.view(B, self.time_L, int(LK/self.time_L))    
        # node-level operation ##################################
        node_in_fft_abs = torch.fft.rfft(node_in, n=self.fft_n ).abs()
        node_in_fft_angle = torch.fft.rfft(node_in, n=self.fft_n ).angle()
        node_in_fft = torch.cat([node_in_fft_abs, node_in_fft_angle], dim=1)
        node_spatial = self.GNN(g, node_in_fft, edge_in)
        phi = self.node_decoder(node_spatial)
        phi = phi / torch.max(torch.abs(phi), dim=0)[0]
        return q, phi # return mode responses and mode shapes


class Model_CFC(nn.Module):   # CFConv, using CPSD as edge features, [FFT.abs, FFT.angle] as node features
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
        self.GNN = CFC(fft_n+2, round((fft_n+2)/2), dim, dim, dropout_rate)
        self.node_decoder = MLP(dim, dim, mode_N, dropout_rate, hid_layer)
        self.time_L = time_L
        self.fft_n = fft_n
    def forward(self, g):
        node_in = g.ndata['acc_Y']
        edge_in = g.edata['cpsd_angle'] / 3.14
        # edge_in = g.edata['cpsd_real']
        # edge_in = torch.sign(g.edata['cpsd_real'])
        # graph-level operation ##################################
        graph_spatial = self.pooling1(g, node_in)
        graph_spatial = self.pooling2(g, graph_spatial)
        [B, LK] = graph_spatial.size()
        q = graph_spatial.view(B, self.time_L, int(LK/self.time_L))    
        # node-level operation ##################################
        node_in_fft_abs = torch.fft.rfft(node_in, n=self.fft_n ).abs()
        node_in_fft_angle = torch.fft.rfft(node_in, n=self.fft_n ).angle()
        node_in_fft = torch.cat([node_in_fft_abs, node_in_fft_angle], dim=1)
        node_spatial = self.GNN(g, node_in_fft, edge_in)
        phi = self.node_decoder(node_spatial)
        phi = phi / torch.max(torch.abs(phi), dim=0)[0]
        return q, phi # return mode responses and mode shapes

class Model_GatedGraphConv(nn.Module):   # GraphSAGE
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
        self.GNN = GatedGraphConv(time_L, time_L, dropout_rate)
        self.node_decoder = MLP(time_L, dim, mode_N, dropout_rate, hid_layer)
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
        node_spatial = self.GNN(g, node_in)
        phi = self.node_decoder(node_spatial)
        phi = phi / torch.max(torch.abs(phi), dim=0)[0]
        return q, phi # return mode responses and mode shapes

def create_model(model_name, dim, time_L, fft_n, mode_N, dropout_rate, hid_layer):
    if model_name == 'SAGE':
        return Model_SAGE(dim, time_L, fft_n, mode_N, dropout_rate, hid_layer)
    elif model_name == 'SAGE2MLP1':
        return Model_2SAGE_1MLP(dim, time_L, fft_n, mode_N, dropout_rate, hid_layer)
    elif model_name == 'SAGE2MLP2':
        return Model_2SAGE_2MLP(dim, time_L, fft_n, mode_N, dropout_rate, hid_layer)
    elif model_name == 'GIN':
        return Model_GIN(dim, time_L, fft_n, mode_N, dropout_rate, hid_layer)
    elif model_name == 'GINE':
        return Model_GINE(dim, time_L, fft_n, mode_N, dropout_rate, hid_layer)
    elif model_name == 'GINE_CPSD':
        return Model_GINE_CPSD(dim, time_L, fft_n, mode_N, dropout_rate, hid_layer)
    elif model_name == 'SAGE_CNN2D':
        return Model_CNN2D(dim, time_L, fft_n, mode_N, dropout_rate, hid_layer)
    elif model_name == 'SAGE_CNN1D':
        return Model_CNN1D(dim, time_L, fft_n, mode_N, dropout_rate, hid_layer)
    elif model_name == 'EdgeGAT':
        return Model_EdgeGAT(dim, time_L, fft_n, mode_N, dropout_rate, hid_layer)
    elif model_name == 'CFC':
        return Model_CFC(dim, time_L, fft_n, mode_N, dropout_rate, hid_layer)
    elif model_name == 'GatedGraphConv':
        return Model_GatedGraphConv(dim, time_L, fft_n, mode_N, dropout_rate, hid_layer)
    else:
        raise ValueError(f"Model {model_name} not recognized.")