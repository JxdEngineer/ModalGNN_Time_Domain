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


# only use node features #####################################################
class Dataset(DGLDataset):
    def __init__(self, graph_ids, time_0, time_L, mode_N,
                 acc_input, phi, node, # node features
                 element, # edge features
                 freq, zeta): # graph features
        self.graph_ids = graph_ids # these definition of ids must be put above the super(), otherwise error happens
        self.time_0 = time_0
        self.time_L = time_L
        self.mode_N = mode_N
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
            graph_sub = graph_sub
            # define graph features ##########################################
            graph_freq = self.freq[graph_id][:self.mode_N].squeeze()
            graph_zeta = self.zeta[graph_id][:self.mode_N].squeeze()
            self.graphs.append(graph_sub)
            self.freqs.append(graph_freq)
            self.zetas.append(graph_zeta)
            print('graph_id =', graph_id)
        # Convert the graph features to tensor type
        self.freqs = torch.tensor(np.array(self.freqs), dtype = torch.float)
        self.zetas = torch.tensor(np.array(self.zetas), dtype = torch.float)
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
    dataset = Dataset(graph_no, time_0, time_L, mode_N,
                 acc_input, phi, node, # node features
                 element, # edge features
                 freq, zeta)  # graph features
    dataloader_dataset = dgl.dataloading.GraphDataLoader(dataset, batch_size=bs,
                                  drop_last=False, shuffle=False,
                                  num_workers=4, pin_memory=True)
    return dataloader_dataset