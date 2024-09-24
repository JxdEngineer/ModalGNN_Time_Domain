# import python lib
import torch
import dgl
from dgl.data import DGLDataset
import numpy as np
import scipy.io as sio # load .mat file

# prepare data 
class Dataset(DGLDataset):
    def __init__(self, graph_ids, time_0, time_L, mode_N, device,
                 acc_input, phi, node, # node features
                 element, element_L, element_theta, # edge features
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
        self.element_L = element_L
        self.element_theta = element_theta
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
            src = np.concatenate((self.element[graph_id][:,0], self.element[graph_id][:,1]), axis=0)-1 # bi-directional edge, left-end node no. (python starts from 0 so minus 1)
            dst = np.concatenate((self.element[graph_id][:,1], self.element[graph_id][:,0]), axis=0)-1 # bi-directional edge, right-end node no.
            graph_sub = dgl.graph((src, dst))  #
            # define node features
            graph_sub.ndata['acc_Y'] = torch.tensor(self.acc_input[graph_id][:, self.time_0:(self.time_0+self.time_L)], dtype = torch.float)  # remove the first time_0 data that is affected by the impact force
            graph_sub.ndata['acc_Y'] = graph_sub.ndata['acc_Y'] / torch.max(torch.abs(graph_sub.ndata['acc_Y'])) # normalization
            graph_sub.ndata['phi_Y'] = torch.tensor(self.phi[graph_id][:, 0:self.mode_N], dtype = torch.float)
            graph_sub.ndata['node'] = torch.tensor(self.node[graph_id], dtype = torch.float)
            node_mask = torch.ones(len(self.node[graph_id]), dtype=torch.bool)
            graph_sub.ndata['mask'] = node_mask
            # define edge features
            edata_L = torch.tensor(self.element_L[graph_id][:, 0], dtype = torch.float)
            graph_sub.edata['L'] = torch.cat((edata_L, edata_L), 0).unsqueeze(1)  # undirectional edge, so double the features
            edata_theta = torch.tensor(self.element_theta[graph_id][:, 0], dtype = torch.float)
            graph_sub.edata['theta'] = torch.cat((edata_theta, edata_theta), 0).unsqueeze(1)  # undirectional edge, so double the features   
            graph_sub = graph_sub.to(self.device)
            # define graph features
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
    element_L = mat_contents['element_L_out'][:, 0]
    element_theta = mat_contents['element_theta_out'][:, 0]
    dataset = Dataset(graph_no, time_0, time_L, mode_N, device, 
                 acc_input, phi, node, # node features
                 element, element_L, element_theta, # edge features
                 freq, zeta)  # graph features
    dataloader_dataset = dgl.dataloading.GraphDataLoader(dataset, batch_size=bs,
                                  drop_last=False, shuffle=True)
    return dataloader_dataset