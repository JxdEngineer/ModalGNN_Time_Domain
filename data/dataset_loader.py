# import python lib
import torch
import dgl
from dgl.data import DGLDataset
import numpy as np
import scipy.io as sio # load .mat file
import scipy.signal as signal
import matplotlib.pyplot as plt
import yaml

# Load config
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# prepare data, use edge features
# class Dataset(DGLDataset):
#     def __init__(self, graph_ids, time_0, time_L, mode_N, device,
#                  acc_input, phi, node, fft_abs, fft_angle, # node features
#                  element, element_L, element_theta, # edge features
#                  freq, zeta): # graph features
#         self.graph_ids = graph_ids # these definition of ids must be put above the super(), otherwise error happens
#         self.time_0 = time_0
#         self.time_L = time_L
#         self.mode_N = mode_N
#         self.device = device
#         self.acc_input = acc_input
#         self.fft_abs = fft_abs
#         self.fft_angle = fft_angle
#         self.phi = phi
#         self.node = node
#         self.element = element
#         self.element_L = element_L
#         self.element_theta = element_theta
#         self.freq = freq
#         self.zeta = zeta
#         super(Dataset, self).__init__(name='ModalGNN')   
#     def process(self):
#         self.graphs = []
#         self.freqs = []
#         self.zetas = []
#         # For each graph ID...
#         for graph_id in self.graph_ids:
#             # Create a graph and add it to the list of graphs and labels.  
#             src = np.concatenate((self.element[graph_id][:,0], self.element[graph_id][:,1]), axis=0) - 1 # bi-directional edge, left-end node no. (python starts from 0 so minus 1)
#             dst = np.concatenate((self.element[graph_id][:,1], self.element[graph_id][:,0]), axis=0) - 1 # bi-directional edge, right-end node no.
#             graph_sub = dgl.graph((src, dst))  #
#             # define node features ##########################################
#             acc_Y = self.acc_input[graph_id][:, self.time_0:(self.time_0+self.time_L)]
#             graph_sub.ndata['acc_Y'] = torch.tensor(acc_Y, dtype = torch.float)  # remove the first time_0 data that is affected by the impact force
#             graph_sub.ndata['acc_Y'] = graph_sub.ndata['acc_Y'] / torch.max(torch.abs(graph_sub.ndata['acc_Y'])) # normalization
#             graph_sub.ndata['phi_Y'] = torch.tensor(self.phi[graph_id][:, 0:self.mode_N], dtype = torch.float)
#             graph_sub.ndata['node'] = torch.tensor(self.node[graph_id], dtype = torch.float)
#             # define node mask
#             node_mask = torch.ones(len(self.node[graph_id]), dtype=torch.bool)
#             graph_sub.ndata['mask'] = node_mask
#             # define acc_Y_PSD
#             # psd_features = []
#             # for node in graph_sub.nodes():
#             #     _, psd = signal.welch(acc_Y[node, :], fs=200, nperseg=256, 
#             #                           nfft=config['model']['fft_n'])
#             #     psd_features.append(psd)
#             # psd_tensor = torch.tensor(psd_features, dtype=torch.float)
#             # graph_sub.ndata['acc_Y_PSD'] = psd_tensor
#             # fig, ax = plt.subplots(2, 1, layout="constrained")
#             # ax[0].plot(graph_sub.ndata['acc_Y_PSD'][14,:])
#             # ax[1].plot(graph_sub.ndata['acc_Y_PSD'][54,:])
#             graph_sub.ndata['acc_Y_FFT_abs'] = torch.tensor(self.fft_abs[graph_id], dtype = torch.float)
#             graph_sub.ndata['acc_Y_FFT_angle'] = torch.tensor(self.fft_angle[graph_id], dtype = torch.float)
            
#             # define edge features ##########################################
#             # graph_sub.edata['element'] = torch.tensor(self.element[graph_id], dtype = torch.float) - 1
#             # edata_L = torch.tensor(self.element_L[graph_id][:, 0], dtype = torch.float)
#             # graph_sub.edata['L'] = torch.cat((edata_L, edata_L), 0).unsqueeze(1)  # undirectional edge, so double the features
#             # edata_theta = torch.tensor(self.element_theta[graph_id][:, 0], dtype = torch.float)
#             # graph_sub.edata['theta'] = torch.cat((edata_theta, edata_theta), 0).unsqueeze(1)  # undirectional edge, so double the features   
#             # # calculate CPSD between adjacent node acc and store them as edge features
#             # edge_features1 = [] # Initialize an empty list to store edge features (CPSD values)
#             # edge_features2 = [] # Initialize an empty list to store edge features (CPSD values)
#             # edge_features3 = [] # Initialize an empty list to store edge features (CPSD values)
#             # for u, v in zip(src, dst): # Iterate over each edge to calculate the CPSD
#             #     # Get time series data for the source and destination nodes
#             #     time_series_u = acc_Y[u]
#             #     time_series_v = acc_Y[v]
#             #     # Calculate the CPSD between the time series of the two nodes
#             #     _, cpsd = signal.csd(time_series_u, time_series_v, 
#             #                          nfft=config['model']['fft_n'], fs=200)
#             #     # Extract real and imaginary parts of CPSD
#             #     edge_feature1 = np.abs(cpsd)
#             #     edge_feature2 = np.angle(cpsd)
#             #     edge_feature3 = np.real(cpsd)
#             #     # Add the edge feature to the list
#             #     edge_features1.append(edge_feature1)   
#             #     edge_features2.append(edge_feature2) 
#             #     edge_features3.append(edge_feature3) 
#             # # Convert edge features to a torch tensor and assign them to the graph
#             # edge_features1_tensor = torch.tensor(edge_features1, dtype=torch.float)
#             # edge_features2_tensor = torch.tensor(edge_features2, dtype=torch.float)
#             # edge_features3_tensor = torch.tensor(edge_features3, dtype=torch.float)
#             # # fig, ax = plt.subplots(2, 1, layout="constrained")
#             # # ax[0].plot(edge_features1_tensor[7,:])
#             # # ax[1].plot(edge_features2_tensor[7,:])
#             # graph_sub.edata['cpsd_abs'] = edge_features1_tensor
#             # graph_sub.edata['cpsd_angle'] = edge_features2_tensor
#             # graph_sub.edata['cpsd_real'] = edge_features3_tensor
            
#             graph_sub = graph_sub.to(self.device)
#             # define graph features ##########################################
#             graph_freq = self.freq[graph_id][:self.mode_N].squeeze()
#             graph_zeta = self.zeta[graph_id][:self.mode_N].squeeze()
#             self.graphs.append(graph_sub)
#             self.freqs.append(graph_freq)
#             self.zetas.append(graph_zeta)
#             print('graph_id =', graph_id)
#         # Convert the graph features to tensor type
#         self.freqs = torch.tensor(self.freqs, dtype = torch.float).to(self.device)
#         self.zetas = torch.tensor(self.zetas, dtype = torch.float).to(self.device)
#     def __getitem__(self, i):
#         return self.graphs[i], self.freqs[i], self.zetas[i]
#     def __len__(self):
#         return len(self.graphs)

# def get_dataset(data_path, bs, graph_no, time_0, time_L, mode_N, device):
#     # Load data
#     mat_contents = sio.loadmat(data_path)
#     acc_input = mat_contents['acceleration_time_out'][:, 0]
#     freq = mat_contents['frequency_out'][:, 0]
#     phi = mat_contents['modeshape_out'][:, 0]   # true mode shape
#     # phi = abs(mat_contents['modeshape_out'][:, 0])*1  # absolute mode shape
#     fft_abs = mat_contents['acceleration_fft_abs_out'][:, 0]
#     fft_angle = mat_contents['acceleration_fft_angle_out'][:, 0]
#     zeta = mat_contents['damping_out'][:, 0]
#     node = mat_contents['node_out'][:, 0]
#     element = mat_contents['element_out'][:, 0]
#     element_L = mat_contents['element_L_out'][:, 0]
#     element_theta = mat_contents['element_theta_out'][:, 0]
#     dataset = Dataset(graph_no, time_0, time_L, mode_N, device, 
#                  acc_input, phi, node, fft_abs, fft_angle, # node features
#                  element, element_L, element_theta, # edge features
#                  freq, zeta)  # graph features
#     dataloader_dataset = dgl.dataloading.GraphDataLoader(dataset, batch_size=bs,
#                                   drop_last=False, shuffle=True)
#     return dataloader_dataset

# only use node features #####################################################
class Dataset(DGLDataset):
    def __init__(self, graph_ids, time_0, time_L, mode_N, device,
                 acc_input, phi, node, # node features
                 element, # edge features
                 freq, zeta): # graph features
        self.graph_ids = graph_ids # these definition of ids must be put above the super(), otherwise error happens
        self.time_0 = time_0
        self.time_L = time_L
        self.mode_N = mode_N
        self.device = device
        self.acc_input = acc_input
        self.phi = phi
        self.node = node
        self.element = element
        self.freq = freq
        self.zeta = zeta
        super(Dataset, self).__init__(name='ModalGNN')   
    def process(self):
        self.graphs = []
        self.freqs = []
        self.zetas = []
        # For each graph ID...
        for graph_id in self.graph_ids:
            # Create a graph and add it to the list of graphs and labels.  
            src = np.concatenate((self.element[graph_id][:,0], self.element[graph_id][:,1]), axis=0) - 1 # bi-directional edge, left-end node no. (python starts from 0 so minus 1)
            dst = np.concatenate((self.element[graph_id][:,1], self.element[graph_id][:,0]), axis=0) - 1 # bi-directional edge, right-end node no.
            graph_sub = dgl.graph((src, dst))  #
            # define node features ##########################################
            acc_Y = self.acc_input[graph_id][:, self.time_0:(self.time_0+self.time_L)]
            graph_sub.ndata['acc_Y'] = torch.tensor(acc_Y, dtype = torch.float)  # remove the first time_0 data that is affected by the impact force
            graph_sub.ndata['acc_Y'] = graph_sub.ndata['acc_Y'] / torch.max(torch.abs(graph_sub.ndata['acc_Y'])) # normalization
            graph_sub.ndata['phi_Y'] = torch.tensor(self.phi[graph_id][:, 0:self.mode_N], dtype = torch.float)
            graph_sub.ndata['node'] = torch.tensor(self.node[graph_id], dtype = torch.float)
            # define node mask
            node_mask = torch.ones(len(self.node[graph_id]), dtype=torch.bool)
            graph_sub.ndata['mask'] = node_mask
            graph_sub = graph_sub.to(self.device)
            # define graph features ##########################################
            graph_freq = self.freq[graph_id][:self.mode_N].squeeze()
            graph_zeta = self.zeta[graph_id][:self.mode_N].squeeze()
            self.graphs.append(graph_sub)
            self.freqs.append(graph_freq)
            self.zetas.append(graph_zeta)
            print('graph_id =', graph_id)
        # Convert the graph features to tensor type
        self.freqs = torch.tensor(self.freqs, dtype = torch.float).to(self.device)
        self.zetas = torch.tensor(self.zetas, dtype = torch.float).to(self.device)
    def __getitem__(self, i):
        return self.graphs[i], self.freqs[i], self.zetas[i]
    def __len__(self):
        return len(self.graphs)

def get_dataset(data_path, bs, graph_no, time_0, time_L, mode_N, device):
    # Load data
    mat_contents = sio.loadmat(data_path)
    acc_input = mat_contents['acceleration_time_out'][:, 0]
    freq = mat_contents['frequency_out'][:, 0]
    phi = mat_contents['modeshape_out'][:, 0]   # true mode shape
    # phi = abs(mat_contents['modeshape_out'][:, 0])*1  # absolute mode shape
    zeta = mat_contents['damping_out'][:, 0]
    node = mat_contents['node_out'][:, 0]
    element = mat_contents['element_out'][:, 0]
    dataset = Dataset(graph_no, time_0, time_L, mode_N, device, 
                 acc_input, phi, node, # node features
                 element, # edge features
                 freq, zeta)  # graph features
    dataloader_dataset = dgl.dataloading.GraphDataLoader(dataset, batch_size=bs,
                                  drop_last=False, shuffle=True)
    return dataloader_dataset