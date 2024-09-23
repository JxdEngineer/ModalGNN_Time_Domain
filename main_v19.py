#%% Training
# v1: learn an estimator for modal identification: input: psd, encoder: MLP
# v3: learn an estimator for modal identification: input: psd, encoder: MLP, only output the first-order mode shape
# v4: input: acc time series, encoder: CNN, output: full-field mode shape (first order)
# v5: input: acc time series, encoder: MLP, output: full-field mode shape (first order)
# v6: input: acc time series, encoder: Transformer, output: full-field mode shape (first order)
# v7: input: acc PSD, encoder: MLP, output: full-field mode shape (first order)
# v8: input: complete/incomplete acc PSD, encoder: MLP, output: full-field mode shape (first order), batch training
#     compare GCN with GraphSAGE
# v10: input: complete/incomplete acc PSD, encoder: MLP, output: full-field mode shape (first three orders), batch training
#     compare GCN with GraphSAGE
# v11: input: complete/incomplete acc PSD, encoder: MLP, output: full-field mode shape (first three orders), batch training
#     compare GCN with GraphSAGE, add edge information
# v12: input: complete/incomplete acc FFT, encoder: MLP, output: full-field mode shape (first three orders), batch training
#     GraphSAGE, add edge information, mode shapes with signs
# v13: use physical information and time-history to identify modal properties
# v14: input: complete/incomplete acc features, encoder: MLP, output: full-field mode shape (first three orders), batch training
#     GraphSAGE, add edge information, absolute mode shapes, use frequency information
# v15: use the loading equation in the frequency domain as the loss function
# v16: use mode superposition in time domain
# v17: improve generalization ability
# v18: add a new loss term that ensures the FFT.abs of modal responses only has one peak. Try different temporal models to encode the time series.
# v19: use edge features

# use the two lines to solve "OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
# OMP: Hint This means that multiple copies of the OpenMP runtime have been linked into the program. That is dangerous, since it can degrade performance or cause incorrect results. The best thing to do is to ensure that only a single OpenMP runtime is linked into the process, e.g. by avoiding static linking of the OpenMP runtime in any library. As an unsafe, unsupported, undocumented workaround you can set the environment variable KMP_DUPLICATE_LIB_OK=TRUE to allow the program to continue to execute, but that may cause crashes or silently produce incorrect results. For more information, please see http://www.intel.com/software/products/support/."
import os
# from sqlite3 import InterfaceError
# os.environ['KMP_DUPLICATE_LIB_OK']='True'

# pytorch lib
import random
import torch
import torch.nn as nn
from torch.nn import Parameter as Param
import torch.nn.init as init
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.nn.init as init
from torch_scatter import scatter_add
from torch.optim.lr_scheduler import StepLR
# geometric lib
import dgl
import dgl.function as fn
import dgl.nn as dglnn
from dgl.nn import EdgeWeightNorm, GraphConv, GINConv
from dgl.dataloading import GraphDataLoader
from dgl.data import DGLDataset
# other lib
import pandas as pd
from tqdm import tqdm
import numpy as np
import math
import matplotlib.pyplot as plt
import time
import copy
# load .mat file
from os.path import dirname, join as pjoin
import scipy.io as sio
from scipy.sparse import csc_matrix, hstack, vstack
from scipy.sparse.linalg import inv
from scipy.linalg import sqrtm
from scipy.stats import kurtosis, skew
from scipy.signal import welch
import networkx as nx
import wandb
# from util import *
# torch.set_default_tensor_type(torch.DoubleTensor)

# fix Seed - DGL cannot reproduce results by fixing the seed
seed = 123
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
dgl.seed(seed)
dgl.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ['CUDNN_DETERMINISTIC'] = '1'
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
torch.set_num_threads(1)

# Load data
mat_contents = sio.loadmat("C:/Users/14360/Desktop/trapezoid_time_bottom_input_100sample_lowpass.mat")
acc_input = mat_contents['acceleration_time_out'][:, 0]
skip_L = 2000 # skip the first skip_L data points because of the unwanted impact response
time_L = 2000 # length of input time series
fft_n = 1024
hid_layer = 0
dropout_rate = 0.2
modeN = 7  # number of modes to be identified

freq = mat_contents['frequency_out'][:, 0]
phi = mat_contents['modeshape_out'][:, 0]   # true mode shape
# phi = abs(mat_contents['modeshape_out'][:, 0])*1  # absolute mode shape
damping = mat_contents['damping_out'][:, 0]
node = mat_contents['node_out'][:, 0]
element = mat_contents['element_out'][:, 0]
element_L = mat_contents['element_L_out'][:, 0]
element_theta = mat_contents['element_theta_out'][:, 0]

# Define a GNN using SAGE
class SAGE(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super().__init__()
        self.conv1 = dglnn.SAGEConv(in_feats, hid_feats, aggregator_type='pool')
        self.conv2 = dglnn.SAGEConv(hid_feats, hid_feats, aggregator_type='pool')
        self.conv3 = dglnn.SAGEConv(hid_feats, out_feats, aggregator_type='pool')
        self.norm = nn.BatchNorm1d(hid_feats)
        self.activation = torch.nn.Tanh()
        self.dropout = nn.Dropout(p=dropout_rate)
    def forward(self, graph, inputs):
        h = self.conv1(graph, inputs) 
        # h = self.norm(h)
        h = self.activation(h)
        h = self.dropout(h)
        for i in range(hid_layer):
            h = self.conv2(graph, h)
            # h = self.norm(h)
            h = self.activation(h)
            h = self.dropout(h)
        h = self.conv3(graph, h)
        return h

# Define a GNN using GINE
class GINE(nn.Module):
    def __init__(self, in_feats, out_feats):
        super().__init__()
        self.conv1 = dglnn.GINEConv(nn.Linear(in_feats, out_feats), learn_eps=True)
    def forward(self, graph, node_inputs, edge_inputs):
        h = self.conv1(graph, node_inputs, edge_inputs)
        return h

class GIN(nn.Module):
    def __init__(self, in_feats, out_feats):
        super().__init__()
        self.conv1 = dglnn.GINConv(nn.Linear(in_feats, out_feats))
    def forward(self, graph, node_inputs):
        h = self.conv1(graph, node_inputs)
        return h

class ChebConv(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super().__init__()
        self.conv1 = dglnn.ChebConv(in_feats, hid_feats, 2)
        self.conv2 = dglnn.ChebConv(hid_feats, hid_feats, 2)
        self.conv3 = dglnn.ChebConv(hid_feats, out_feats, 2)
        self.activation = torch.nn.Tanh()
        self.dropout = nn.Dropout(p=dropout_rate)
    def forward(self, graph, inputs):
        h = self.conv1(graph, inputs) 
        # for i in range(hid_layer):
        #     h = self.conv2(graph, h)
        # h = self.conv3(graph, h)
        return h

# Define a MLP
class MLP(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()
        self.layer1 = nn.Linear(in_dim, hid_dim)
        self.layer2 = nn.Linear(hid_dim, hid_dim)
        self.layer3 = nn.Linear(hid_dim, out_dim)
        self.activation = torch.nn.Tanh()
        self.dropout = nn.Dropout(p=dropout_rate)
    def forward(self, x):
        y = self.layer1(x)
        y = self.activation(y)
        y = self.dropout(y)
        for i in range(hid_layer):
            y = self.layer2( y)
            y = self.activation(y)
            y = self.dropout(y)
        y = self.layer3(y)
        return y


# Define the entire model
# class Model(nn.Module):   # GraphSAGE
#     def __init__(self, dim):
#         super().__init__()
#         # use built-in Transformer ##################################
#         self.pooling1 = dglnn.SetTransformerEncoder(time_L, n_heads=4, 
#                                                     d_head=4, d_ff=256,
#                                                     n_layers=2)
#         self.pooling2 = dglnn.SetTransformerDecoder(time_L, num_heads=4, 
#                                                     d_head=4, d_ff=256,
#                                                     n_layers=2,
#                                                     k=modeN)       
#         self.GNN = SAGE(fft_n+2, dim, dim)
#         self.node_decoder = MLP(dim, dim, modeN)
#     def forward(self, g):
#         node_in = g.ndata['acc_Y']
#         # graph-level operation ##################################
#         graph_spatial = self.pooling1(g, node_in)
#         graph_spatial = self.pooling2(g, graph_spatial)
#         [B, LK] = graph_spatial.size()
#         q = graph_spatial.view(B, time_L, int(LK/time_L))    
#         # node-level operation ##################################
#         node_in_fft_abs = torch.fft.rfft(node_in, n=fft_n).abs()
#         node_in_fft_angle = torch.fft.rfft(node_in, n=fft_n).angle()
#         node_in_fft = torch.cat([node_in_fft_abs, node_in_fft_angle], dim=1)
#         node_spatial = self.GNN(g, node_in_fft)
#         phi = self.node_decoder(node_spatial)
#         phi = phi / torch.max(torch.abs(phi), dim=0)[0]
#         return q, phi # return mode responses and mode shapes

# class Model(nn.Module):   # GINE
#     def __init__(self, dim):
#         super().__init__()
#         # use built-in Transformer ##################################
#         self.pooling1 = dglnn.SetTransformerEncoder(time_L, n_heads=4, 
#                                                     d_head=4, d_ff=256,
#                                                     n_layers=2)
#         self.pooling2 = dglnn.SetTransformerDecoder(time_L, num_heads=4, 
#                                                     d_head=4, d_ff=256,
#                                                     n_layers=2,
#                                                     k=modeN)       
#         self.GNN = GINE(fft_n+2, dim)
#         self.edge_encoder = MLP(2, dim, fft_n+2)
#         self.node_decoder = MLP(dim, dim, modeN)
#     def forward(self, g):
#         node_in = g.ndata['acc_Y']
#         edge_in = torch.cat([g.edata['L'], g.edata['theta']], dim=1)
#         # graph-level operation ##################################
#         graph_spatial = self.pooling1(g, node_in)
#         graph_spatial = self.pooling2(g, graph_spatial)
#         [B, LK] = graph_spatial.size()
#         q = graph_spatial.view(B, time_L, int(LK/time_L))    
#         # node-level operation ##################################
#         node_in_fft_abs = torch.fft.rfft(node_in, n=fft_n).abs()
#         node_in_fft_angle = torch.fft.rfft(node_in, n=fft_n).angle()
#         node_in_fft = torch.cat([node_in_fft_abs, node_in_fft_angle], dim=1)
#         edge_in_encoded = self.edge_encoder(edge_in)
#         node_spatial = self.GNN(g, node_in_fft, edge_in_encoded)
#         phi = self.node_decoder(node_spatial)
#         phi = phi / torch.max(torch.abs(phi), dim=0)[0]
#         return q, phi # return mode responses and mode shapes

# class Model(nn.Module):   # GIN
#     def __init__(self, dim):
#         super().__init__()
#         # use built-in Transformer ##################################
#         self.pooling1 = dglnn.SetTransformerEncoder(time_L, n_heads=4, 
#                                                     d_head=4, d_ff=256,
#                                                     n_layers=2)
#         self.pooling2 = dglnn.SetTransformerDecoder(time_L, num_heads=4, 
#                                                     d_head=4, d_ff=256,
#                                                     n_layers=2,
#                                                     k=modeN)       
#         self.GNN = GIN(fft_n+2, dim)
#         self.node_decoder = MLP(dim, dim, modeN)
#     def forward(self, g):
#         node_in = g.ndata['acc_Y']
#         # graph-level operation ##################################
#         graph_spatial = self.pooling1(g, node_in)
#         graph_spatial = self.pooling2(g, graph_spatial)
#         [B, LK] = graph_spatial.size()
#         q = graph_spatial.view(B, time_L, int(LK/time_L))    
#         # node-level operation ##################################
#         node_in_fft_abs = torch.fft.rfft(node_in, n=fft_n).abs()
#         node_in_fft_angle = torch.fft.rfft(node_in, n=fft_n).angle()
#         node_in_fft = torch.cat([node_in_fft_abs, node_in_fft_angle], dim=1)
#         node_spatial = self.GNN(g, node_in_fft)
#         phi = self.node_decoder(node_spatial)
#         phi = phi / torch.max(torch.abs(phi), dim=0)[0]
#         return q, phi # return mode responses and mode shapes

class Model(nn.Module):   # ChebConv
    def __init__(self, dim):
        super().__init__()
        # use built-in Transformer ##################################
        self.pooling1 = dglnn.SetTransformerEncoder(time_L, n_heads=4, 
                                                    d_head=4, d_ff=256,
                                                    n_layers=2)
        self.pooling2 = dglnn.SetTransformerDecoder(time_L, num_heads=4, 
                                                    d_head=4, d_ff=256,
                                                    n_layers=2,
                                                    k=modeN)       
        # self.GNN = ChebConv(fft_n+2, dim, dim)
        self.GNN = ChebConv(2000, dim, dim)
        self.node_decoder = MLP(dim, dim, modeN)
    def forward(self, g):
        node_in = g.ndata['acc_Y']
        # graph-level operation ##################################
        graph_spatial = self.pooling1(g, node_in)
        graph_spatial = self.pooling2(g, graph_spatial)
        [B, LK] = graph_spatial.size()
        q = graph_spatial.view(B, time_L, int(LK/time_L))    
        # node-level operation ##################################
        # node_in_fft_abs = torch.fft.rfft(node_in, n=fft_n).abs()
        # node_in_fft_angle = torch.fft.rfft(node_in, n=fft_n).angle()
        # node_in_fft = torch.cat([node_in_fft_abs, node_in_fft_angle], dim=1)
        node_spatial = self.GNN(g, node_in)
        phi = self.node_decoder(node_spatial)
        phi = phi / torch.max(torch.abs(phi), dim=0)[0]
        return q, phi # return mode responses and mode shapes

# Use GPU for training
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# prepare data 
class Dataset(DGLDataset):
    def __init__(self, graph_ids, time0):
        self.graph_ids = graph_ids # these definition of ids must be put above the super(), otherwise error happens
        self.time0 = time0
        super(Dataset, self).__init__(name='ModalGNN')   
    def process(self):
        self.graphs = []
        self.labels = []
        # For each graph ID...
        for graph_id in self.graph_ids:
            # Create a graph and add it to the list of graphs and labels.  
            src = np.concatenate((element[graph_id][:,0], element[graph_id][:,1]), axis=0)-1 # bi-directional edge, left-end node no. (python starts from 0 so minus 1)
            dst = np.concatenate((element[graph_id][:,1], element[graph_id][:,0]), axis=0)-1 # bi-directional edge, right-end node no.
            graph_sub = dgl.graph((src, dst))  #
            # define node features
            graph_sub.ndata['acc_Y'] = torch.tensor(acc_input[graph_id][:, self.time0:(self.time0+time_L)], dtype = torch.float)  # remove the first time0 data that is affected by the impact force
            graph_sub.ndata['acc_Y'] = graph_sub.ndata['acc_Y'] / torch.max(torch.abs(graph_sub.ndata['acc_Y'])) # normalization
            # graph_sub.ndata['acc_Y'] = torch.nn.functional.normalize(graph_sub.ndata['acc_Y']) # normalization
            graph_sub.ndata['phi_Y'] = torch.tensor(phi[graph_id][:, 0:modeN], dtype = torch.float)        
            # define edge features
            edata_L = torch.tensor(element_L[graph_id][:, 0], dtype = torch.float)
            graph_sub.edata['L'] = torch.cat((edata_L, edata_L), 0).unsqueeze(1)  # undirectional edge, so double the features
            edata_theta = torch.tensor(element_theta[graph_id][:, 0], dtype = torch.float)
            graph_sub.edata['theta'] = torch.cat((edata_theta, edata_theta), 0).unsqueeze(1)  # undirectional edge, so double the features
            # feature propagation
            node_mask = torch.ones(len(node[graph_id]), dtype=torch.bool)
            # missing_indices = np.array(range(1, len(node[graph_id]), 2))
            # node_mask[missing_indices] = False
            # missing_indices = np.array(range(1, len(node[graph_id]), 3))
            # node_mask[missing_indices] = False
            # missing_indices = np.array(range(2, len(node[graph_id]), 3))
            # node_mask[missing_indices] = False
            # missing_indices = np.array(range(3, len(node[graph_id]), 3))
            # node_mask[missing_indices] = False
            missing_ratio = np.count_nonzero(node_mask == False)/len(node_mask)
            print('missing_ratio =', missing_ratio)
            edge_index =  torch.LongTensor([list(row) for row in list(zip(src, dst))]).T
            acc_complete = graph_sub.ndata['acc_Y']
            acc_FP = Feature_Propagation(acc_complete, node_mask, edge_index, len(node[graph_id]))
            acc_FP[acc_complete[:, 1]==0, :] = 0 # reponses of constrained DOFs should be zero
            graph_sub.ndata['acc_Y'] = acc_FP
            graph_sub.ndata['mask'] = node_mask
            
            # g = graph_sub
            label = 0
            self.graphs.append(graph_sub.to(device))
            self.labels.append(label)
        # Convert the label list to tensor for saving.
        self.labels = torch.LongTensor(self.labels)
    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]
    def __len__(self):
        return len(self.graphs)

# Define the MAC funcion
def MAC(x, y):
    assert x.shape == y.shape, "Mode shapes must have the same shape"
    numerator = np.abs(np.dot(x, y.T))**2
    denominator_x = np.dot(x, x.T)
    denominator_y = np.dot(y, y.T)
    mac = numerator / (denominator_x * denominator_y)
    return mac

# Define the Feature Propagation Func to fill missing nodal features (DOFs without measurements)
def get_propagation_matrix(edge_index, n_nodes):
    # Initialize all edge weights to ones if the graph is unweighted)
    """
    Given an edge_index, return the same edge_index and edge weights computed as
    \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}} \mathbf{\hat{D}}^{-1/2}.
    """
    edge_weight = torch.ones((edge_index.size(1),), device=edge_index.device)
    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, col, dim=0, dim_size=n_nodes)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0)
    DAD = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col] 
    adj = torch.sparse.FloatTensor(edge_index, values=DAD, size=(n_nodes, n_nodes)).to(edge_index.device)
    return adj

def Feature_Propagation(x, mask, edge_index, n_nodes):
    # out is inizialized to 0 for missing values. However, its initialization does not matter for the final
    # value at convergence
    out = x
    if mask is not None:
        out = torch.zeros_like(x)
        out[mask] = x[mask]
    num_iterations = 30
    adj = get_propagation_matrix(edge_index, n_nodes)
    for _ in range(num_iterations):
        # Diffuse current features
        out = torch.sparse.mm(adj, out)
        # Reset original known features
        out[mask] = x[mask]
    return out

# loss function using fft of modal responses
def loss_terms(q_pred, phi_pred, graph):
    graph_unbatched = dgl.unbatch(graph[0])
    phi_index1 = 0
    phi_index2 = 0
    
    for i in range(len(graph_unbatched)):
        phi_index1 = phi_index2
        phi_index2 = phi_index1 + dgl.DGLGraph.number_of_nodes(graph_unbatched[i])
        q_pred_unbatched = q_pred[i, :, :]
        q_pred_unbatched_fft = torch.fft.rfft(q_pred_unbatched.T, n=fft_n).abs()
        phi_pred_unbatched = phi_pred[phi_index1:phi_index2]
        if i == 0:
            acc_pred = phi_pred_unbatched @ q_pred_unbatched.T
            q_corr = torch.corrcoef(q_pred_unbatched.T)
            q_fft_corr = torch.corrcoef(q_pred_unbatched_fft)
        else:
            acc_pred = torch.cat((acc_pred, phi_pred_unbatched @ q_pred_unbatched.T), 0)
            q_corr = torch.block_diag(q_corr, torch.corrcoef(q_pred_unbatched.T))
            q_fft_corr = torch.block_diag(q_fft_corr, torch.corrcoef(q_pred_unbatched_fft))  
    batched_eye = torch.eye(modeN*len(graph_unbatched), device=device)
    acc_true = graph[0].ndata['acc_Y']
    # incomplete measurements
    node_mask = graph[0].ndata['mask'] 
    acc_true = acc_true[node_mask, :]
    acc_pred = acc_pred[node_mask, :]
    # calculate loss
    loss1 = loss_func(acc_pred, acc_true)
    loss2 = loss_func(q_corr, batched_eye)
    loss3 = loss_func(q_fft_corr, batched_eye)
    
    # loss4 = torch.zeros_like(loss1) # unimodal loss
    loss4 = loss_func(phi_pred, graph[0].ndata['phi_Y'])
    # loss1 = torch.zeros_like(loss4)
    # loss2 = torch.zeros_like(loss4)
    # loss3 = torch.zeros_like(loss4)
    # loss4 = loss_max1(phi_pred) * c4
    # loss5 = loss_func(phi_pred[boundary_mask], torch.zeros_like(phi_pred[boundary_mask])) * c5 
    return loss1, loss2, loss3, loss4

# kk = int(N_train/N_valid)+1
# valid_no = np.array(range(0, N_valid)) + (kk-1)*N_valid
# train_no = np.setdiff1d(np.array(range(0,N_train+N_valid)), valid_no)
# test_no = np.array(range(N_train+N_valid,N_all))

train_no = np.array(range(0, 10))
# train_no = np.concatenate((np.array(range(0, 40)), np.array(range(50, 100))))
valid_no = np.array(range(95, 100))

train_set = Dataset(graph_ids=train_no, time0=skip_L)
valid_set = Dataset(graph_ids=valid_no, time0=skip_L)
# test_set = Dataset(graph_ids=test_no)

model_dim = 64
model = Model(dim = model_dim)

bs = 32
dataloader_train = dgl.dataloading.GraphDataLoader(train_set, batch_size=bs,
                              drop_last=False, shuffle=True)
dataloader_valid = dgl.dataloading.GraphDataLoader(valid_set, batch_size=bs,
                              drop_last=False, shuffle=True)
print('done')
#%% Train model
plt.close('all')
model.to(device)
loss_func = nn.MSELoss()
# loss_func = nn.L1Loss()
# loss_func = nn.SmoothL1Loss()
# loss_func_cos = nn.CosineSimilarity(dim=0)

learning_rate = 0.0005
n_epoch = 2000
opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = StepLR(opt, step_size=100, gamma=1)

# opt = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
# scheduler = torch.optim.lr_scheduler.CyclicLR(opt, base_lr=learning_rate, max_lr=0.001)

start_time = time.time()

loss_train_meter = []
loss1_train_meter = []
loss2_train_meter = []
loss3_train_meter = []
loss4_train_meter = []
loss5_train_meter = []

loss_valid_meter = []
loss1_valid_meter = []
loss2_valid_meter = []
loss3_valid_meter = []
loss4_valid_meter = []
loss5_valid_meter = []

# start a new wandb run to track this script #######################
# wandb.init(
#     # set the wandb project where this run will be logged
#     project="ModalTemporalGNN",
#     # track hyperparameters and run metadata
#     config={
#     "N_train": len(train_no),
#     "time_L": time_L,
#     "learning rate": learning_rate,
#     "model dim": model_dim,
#     "dropout rate": dropout_rate,
#     "N_mode": modeN,
#     "N_hid_layer": hid_layer,
#     "loss_func": "MSE",
#     "activation_func": "Tanh",
#     "mode_type": "GNN"
#     }
# )

# update W&B config
# api = wandb.Api()
# run = api.run("jxd-engineer/ModalTemporalGNN/3jvvrp2h")
# run.config["N_mode"] = 7
# run.update()

model.train()

c1 = 1
c2 = 1
c3 = 1
c4 = 0
c5 = 0

for epoch in range(n_epoch):
    # model validation ##################################
    epoch_loss_valid = 0
    epoch_loss3_valid = 0
    for graph_valid in dataloader_valid:
        q_pred_valid, phi_pred_valid = model(graph_valid[0])  # model inference
        # phi_pred_valid = graph_valid[0].ndata['phi_Y']  # use true phi for training 
        loss1_valid, loss2_valid, loss3_valid, loss4_valid = loss_terms(q_pred_valid, phi_pred_valid, graph_valid)
        loss_valid = loss1_valid*c1 + loss2_valid*c2 + loss3_valid*c3 + loss4_valid*c4
        epoch_loss3_valid += loss3_valid
        epoch_loss_valid += loss_valid
    epoch_loss3_valid /= len(dataloader_valid)
    epoch_loss_valid /= len(dataloader_valid)
    loss_valid_meter.append(epoch_loss_valid.detach().cpu().numpy())
    loss1_valid_meter.append(loss1_valid.detach().cpu().numpy())
    loss2_valid_meter.append(loss2_valid.detach().cpu().numpy())
    loss3_valid_meter.append(loss3_valid.detach().cpu().numpy())
    loss4_valid_meter.append(loss4_valid.detach().cpu().numpy())
    # loss5_valid_meter.append(loss5_valid.detach().cpu().numpy())
    
    # model training ##################################
    epoch_loss_train = 0
    for graph_train in dataloader_train:
        q_pred_train, phi_pred_train = model(graph_train[0])  # model inference
        # phi_pred_train = graph_train[0].ndata['phi_Y']  # use true phi for training 
        loss1_train, loss2_train, loss3_train, loss4_train = loss_terms(q_pred_train, phi_pred_train, graph_train)
        loss_train = loss1_train*c1 + loss2_train*c2 + loss3_train*c3 + loss4_train*c4
        epoch_loss_train += loss_train
    epoch_loss_train /= len(dataloader_train)
    loss_train_meter.append(epoch_loss_train.detach().cpu().numpy())
    loss1_train_meter.append(loss1_train.detach().cpu().numpy())
    loss2_train_meter.append(loss2_train.detach().cpu().numpy())
    loss3_train_meter.append(loss3_train.detach().cpu().numpy())
    loss4_train_meter.append(loss4_train.detach().cpu().numpy())
    # loss5_train_meter.append(loss5_train.detach().cpu().numpy())
    
    # backpropogation ##################################
    opt.zero_grad()
    epoch_loss_train.backward()
    opt.step()
    scheduler.step()

    if epoch % 20 == 0:
        print('epoch: {}, loss_train: {:.10f}, loss_valid: {:.10f}' .format(epoch, epoch_loss_train, epoch_loss_valid))
        
    # log metrics to wandb #########################
    # wandb.log({"loss_train": epoch_loss_train, "loss_valid": epoch_loss_valid,
    #           "loss3_valid": epoch_loss3_valid})
train_time = (time.time() - start_time)
print("--- %s seconds ---" % train_time)

# [optional] finish the wandb run, necessary in notebooks
# wandb.finish()
# %% plot results
plt.close('all')
# plot log loss curve
plt.figure()
plt.semilogy(loss_train_meter, label='train loss: '+f"{loss_train_meter[-1]:.6f}")
plt.semilogy(loss_valid_meter, label='valid loss: '+f"{loss_valid_meter[-1]:.6f}")
plt.ylabel('MSE loss', fontsize=14)
plt.xlabel('Epoch', fontsize=14)
title_text = "Time={:.3f}".format(train_time)
plt.title(title_text)
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(1, 2, layout="constrained")
# plot different terms in the training loss
ax[0].semilogy(loss1_train_meter, label='Term 1: '+f"{loss1_train_meter[-1]:.6f}")
ax[0].semilogy(loss2_train_meter, label='Term 2: '+f"{loss2_train_meter[-1]:.6f}")
ax[0].semilogy(loss3_train_meter, label='Term 3: '+f"{loss3_train_meter[-1]:.6f}")
ax[0].semilogy(loss4_train_meter, label='Term 4: '+f"{loss4_train_meter[-1]:.5f}")
# ax[0].semilogy(loss5_train_meter, label='Term 5: '+f"{loss5_train_meter[-1]:.5f}")
title_text = "Training loss"
ax[0].set_title(title_text)
ax[0].set_ylabel('log(Loss)', fontsize=14)
ax[0].set_xlabel('Epoch', fontsize=14)
ax[0].set_ylim([0.000001, 10])
ax[0].grid()
ax[0].legend()
# plot different terms in the validation loss
ax[1].semilogy(loss1_valid_meter, label='Term 1: '+f"{loss1_valid_meter[-1]:.6f}")
ax[1].semilogy(loss2_valid_meter, label='Term 2: '+f"{loss2_valid_meter[-1]:.6f}")
ax[1].semilogy(loss3_valid_meter, label='Term 3: '+f"{loss3_valid_meter[-1]:.6f}")
ax[1].semilogy(loss4_valid_meter, label='Term 4: '+f"{loss4_valid_meter[-1]:.5f}")
# ax[1].semilogy(loss5_valid_meter, label='Term 5: '+f"{loss5_valid_meter[-1]:.5f}")
title_text = "Validation loss"
ax[1].set_title(title_text)
ax[1].set_ylabel('log(Loss)', fontsize=14)
ax[1].set_xlabel('Epoch', fontsize=14)
ax[1].set_ylim([0.000001, 10])
ax[1].grid()
ax[1].legend()
# %% save model
# PATH = "GNN_time_domain.pt"
# torch.save(model.state_dict(), PATH)
# %% test trained model - single sample
def ModalId(f, Pxx):
    # Identify peak frequency (damped natural frequency)
    peak_index = np.argmax(Pxx)
    f_d = f[peak_index]
    # Find half-power points
    half_power_value = Pxx[peak_index] / math.sqrt(2)
    left_idx = np.where(Pxx[:peak_index] <= half_power_value)[0][-1]
    right_idx = np.where(Pxx[peak_index:] <= half_power_value)[0][0] + peak_index
    delta_f = f[right_idx] - f[left_idx]
    # Calculate damping ratio
    zeta = delta_f / (2 * f_d)
    # Calculate natural frequency
    # f_n = f_d / np.sqrt(1 - zeta**2)
    f_n = f_d
    return f_n, zeta

# PATH = "GNN_time_domain.pt"
# model.load_state_dict(torch.load(PATH))
plt.close('all')
model.eval()
model = model.cpu()

test_no = 7 - 1 # number of the tested truss
test_data = Dataset(graph_ids = [test_no], time0=skip_L*1)[0]

graph_test = test_data[0].to('cpu')
node_mask = graph_test.ndata['mask']

node_test = node[test_no]
element_test = element[test_no] - 1
freq_test = freq[test_no]
damping_test = damping[test_no]

q_pred_test, phi_pred_test = model(graph_test)
# phi_pred_test = graph_test.ndata['phi_Y']  # use true phi for testing
q_pred_test = torch.squeeze(q_pred_test, 0)


acc_pred = phi_pred_test @ q_pred_test.T
acc_true = graph_test.ndata['acc_Y']

# plot acc time series and PSD #########################################
dof_no = np.array([20, 25, 30, 35, 40]) - 1  
fig, ax = plt.subplots(5, 2, layout="constrained")
title_text = "Truss No.={:.0f}".format(test_no+1)
fig.suptitle(title_text, fontsize=16)
for i in range(5):
    ax[i, 0].plot(acc_true[dof_no[i], :].detach().numpy(), linestyle='--', label='True')
    ax[i, 0].plot(acc_pred[dof_no[i], :].detach().numpy(), label='Pred')
    ax[i, 0].set_ylabel('Acc', fontsize=14)
    ax[i, 0].grid()
    title_text = "DOF={:.0f}".format(dof_no[i]+1)
    ax[i, 0].set_title(title_text, fontsize=14)
    ax[i, 0].legend()
    
    frequencies, psd_pred = welch(acc_pred[dof_no[i], :].to('cpu').detach().numpy(),
                             fs=200, nperseg=256, nfft=fft_n)
    frequencies, psd_true = welch(acc_true[dof_no[i], :].to('cpu').detach().numpy(),
                             fs=200, nperseg=256, nfft=fft_n)
    ax[i, 1].plot(frequencies, psd_true, linestyle='--', label='True')
    ax[i, 1].plot(frequencies, psd_pred, label='Pred')
    ax[i, 1].set_xlabel('Frequency [Hz]', fontsize=14)
    ax[i, 1].set_ylabel('PSD', fontsize=14)
    ax[i, 1].grid(True)
    ax[i, 1].legend()
    ax[i, 1].set_xlim(0, 100)
    for j in range(5):
        ax[i, 1].plot([freq_test[j], freq_test[j]], [0, max(psd_pred)], color='#FF1F5B')   

# plot PSD of modal responses #########################################
fig, ax = plt.subplots(modeN, 4, layout="constrained")
title_text = "Truss No.={:.0f}".format(test_no+1)
fig.suptitle(title_text, fontsize=16)
for mode_no in range(modeN):
    ax[mode_no, 0].plot(q_pred_test[:, mode_no].to('cpu').detach().numpy())
    ax[mode_no, 0].set_ylabel('Modal acc')
    ax[mode_no, 0].grid()
    
    frequencies, psd = welch(q_pred_test[:, mode_no].to('cpu').detach().numpy(),
                             fs=200, nperseg=256, nfft=fft_n)
    # [f_n, zeta] = ModalId(frequencies, psd)
    ax[mode_no, 1].plot(frequencies, psd)
    # title_text = "component={:.0f}, f={:.3f}, zeta={:.5}".format(mode_no+1, f_n, zeta)
    # ax[mode_no, 1].set_title(title_text)
    ax[mode_no, 1].set_xlabel('Frequency [Hz]')
    ax[mode_no, 1].set_ylabel('PSD')
    ax[mode_no, 1].grid(True)
    ax[mode_no, 1].set_xlim(0, 50)
    for i in range(5):
        ax[mode_no, 1].plot([freq_test[i], freq_test[i]], [0, max(psd)], color='#FF1F5B')    
    
    phi_pred = phi_pred_test[:, mode_no].to('cpu').detach().numpy().squeeze()
    # phi_pred = phi_pred / max(abs(phi_pred)) * 3
    phi_pred = phi_pred * 3
    node_pred = np.zeros([len(node_test), 2])
    node_pred[:, 0] = node_test[:, 0]
    node_pred[:, 1] = node_test[:, 1] + phi_pred
    
    phi_true = graph_test.ndata['phi_Y'][:, mode_no].to('cpu').detach().numpy().squeeze() * 3
    node_true = np.zeros([len(node_test), 2])
    node_true[:, 0] = node_test[:, 0]
    node_true[:, 1] = node_test[:, 1] + phi_true
    
    for ele in element_test:
        node1 = node_test[ele[0]]
        node2 = node_test[ele[1]]
        ax[mode_no, 3-1].plot([node1[0], node2[0]], [node1[1], node2[1]], 'k--')
    # ax[mode_no, 2].plot(node_true[:, 0], node_true[:, 1], 'ko', markersize=7, label='undeformed')        
    for ele in element_test:
        node1 = node_pred[ele[0]]
        node2 = node_pred[ele[1]]
        ax[mode_no, 3-1].plot([node1[0], node2[0]], [node1[1], node2[1]], 'b-')
        
    # ax[mode_no, 2].plot(node_pred[:, 0], node_pred[:, 1], 'bo', markersize=7, label='predicted')
    
    ax[mode_no, 3-1].plot(node_pred[:, 0], node_pred[:, 1], 'o', markersize=3, label='identified_known', color='#AF58BA')
    ax[mode_no, 3-1].plot(node_pred[~node_mask, 0], node_pred[~node_mask, 1], 's', markersize=3, label='identified_unknown', color='#00CD6C')
    ax[mode_no, 3-1].grid()
    ax[mode_no, 3-1].set_ylabel('shape_id')
    
    # ax[mode_no, 2].hist(q_pred_test[:, mode_no].to('cpu').detach().numpy(), bins=100, edgecolor='black')
    # ax[mode_no, 2].set_title('Histogram of Modal Responses')
    # ax[mode_no, 2].set_xlabel('Value')
    # ax[mode_no, 2].set_ylabel('Frequency')
    # ax[mode_no, 2].grid(True)
    
    for ele in element_test:
        node1 = node_test[ele[0]]
        node2 = node_test[ele[1]]
        ax[mode_no, 4-1].plot([node1[0], node2[0]], [node1[1], node2[1]], 'k--')
    # ax[mode_no, 3].plot(node_true[:, 0], node_true[:, 1], 'ko', markersize=7, label='undeformed')    
    for ele in element_test:
        node1 = node_true[ele[0]]
        node2 = node_true[ele[1]]
        ax[mode_no, 4-1].plot([node1[0], node2[0]], [node1[1], node2[1]], 'r-')
    ax[mode_no, 4-1].plot(node_true[:, 0], node_true[:, 1], 'ro', markersize=3, label='true')
    # ax[mode_no, 2].set_aspect('equal')
    ax[mode_no, 4-1].grid()
    # ylabel_text = "shape={:.0f}".format(mode_no+1)
    ax[mode_no, 4-1].set_ylabel('shape_true')
    title_text = "mode={:.0f}, f={:.3f}, zeta={:.4f}".format(mode_no+1, freq_test[mode_no, 0], damping_test[mode_no, 0])
    ax[mode_no, 4-1].set_title(title_text, fontsize=12)  
       
# match identified mode shapes with true mode shapes #########################################
# mode_list = [1, 2, 3, 4, 5]
# mode_list = [x - 1 for x in mode_list]
# fig, ax = plt.subplots(5, 2, layout="constrained")
# title_text = "Truss No.={:.0f}".format(test_no+1)
# fig.suptitle(title_text, fontsize=16)
# for mode_no in range(5):
#     phi_pred = phi_pred_test[:, mode_list[mode_no]].to('cpu').detach().numpy().squeeze()
#     phi_pred = phi_pred * 3 / np.sign(phi_pred[10])
#     node_pred = np.zeros([len(node_test), 2])
#     node_pred[:, 0] = node_test[:, 0]
#     node_pred[:, 1] = node_test[:, 1] + phi_pred
#     phi_true = graph_test.ndata['phi_Y'][:, mode_no].to('cpu').detach().numpy().squeeze()
#     phi_true = phi_true  * 3 / np.sign(phi_true[10])
    
#     node_true = np.zeros([len(node_test), 2])
#     node_true[:, 0] = node_test[:, 0]
#     node_true[:, 1] = node_test[:, 1] + phi_true
    
#     for ele in element_test:
#         node1 = node_test[ele[0]]
#         node2 = node_test[ele[1]]
#         ax[mode_no, 0].plot([node1[0], node2[0]], [node1[1], node2[1]], 'k--')
#     # ax[mode_no, 2].plot(node_true[:, 0], node_true[:, 1], 'ko', markersize=7, label='undeformed')        
#     for ele in element_test:
#         node1 = node_true[ele[0]]
#         node2 = node_true[ele[1]]
#         ax[mode_no, 0].plot([node1[0], node2[0]], [node1[1], node2[1]], 'r--')
#     ax[mode_no, 0].plot(node_true[:, 0], node_true[:, 1], 'ro', label='true')
#     ax[mode_no, 0].grid()
#     ax[mode_no, 0].set_ylabel('Mode shape')
#     mac_value = MAC(phi_true, phi_pred)
#     title_text = "mode={:.0f}, f={:.3f}, MAC={:.3f}".format(mode_no+1, freq_test[mode_no, 0], mac_value)
#     ax[mode_no, 0].set_title(title_text, fontsize=14)  
    
#     for ele in element_test:
#         node1 = node_pred[ele[0]]
#         node2 = node_pred[ele[1]]
#         ax[mode_no, 0].plot([node1[0], node2[0]], [node1[1], node2[1]], 'b-')
#     # ax[mode_no, 0].plot(node_pred[:, 0], node_pred[:, 1], 'bo', markersize=7, label='predicted')
#     ax[mode_no, 0].plot(node_pred[:, 0], node_pred[:, 1], 'o', label='identified_known', color='#AF58BA')
#     ax[mode_no, 0].plot(node_pred[~node_mask, 0], node_pred[~node_mask, 1], 's', label='identified_unknown', color='#00CD6C')
#     ax[mode_no, 0].legend()