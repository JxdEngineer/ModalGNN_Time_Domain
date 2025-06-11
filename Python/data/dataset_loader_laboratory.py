# import python lib
import torch
from torch_scatter import scatter_add
import dgl
from dgl.data import DGLDataset
import numpy as np
import scipy.io as sio # load .mat file
import yaml

# Load config
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

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
    num_iterations = 50
    adj = get_propagation_matrix(edge_index, n_nodes)
    for _ in range(num_iterations):
        # Diffuse current features
        out = torch.sparse.mm(adj, out)
        # Reset original known features
        out[mask] = x[mask]
    return out


# only use node features #####################################################
class Dataset(DGLDataset):
    def __init__(self, graph_ids, time_0_list, time_L, mode_N,
                 acc_input, phi, node, # node features
                 element, # edge features
                 freq): # graph features
        self.graph_ids = graph_ids # these definition of ids must be put above the super(), otherwise error happens
        self.time_0_list = time_0_list
        self.time_L = time_L
        self.mode_N = mode_N
        self.acc_input = acc_input
        self.phi = phi
        self.node = node
        self.element = element
        self.freq = freq
        super(Dataset, self).__init__(name='ModalGNN')   
    def process(self):
        self.graphs = []
        self.freqs = []
        # For each graph ID...
        for graph_id in self.graph_ids:
            for time_0 in self.time_0_list:
                # Create a graph and add it to the list of graphs and labels.  
                src = np.concatenate((self.element[graph_id][:,0], self.element[graph_id][:,1]), axis=0) - 1 # bi-directional edge, left-end node no. (python starts from 0 so minus 1)
                dst = np.concatenate((self.element[graph_id][:,1], self.element[graph_id][:,0]), axis=0) - 1 # bi-directional edge, right-end node no.
                graph_sub = dgl.graph((src, dst))  #
                # define node features ##########################################
                acc_Y = self.acc_input[graph_id][:, time_0:(time_0 + self.time_L)]
                graph_sub.ndata['acc_Y'] = torch.tensor(acc_Y, dtype = torch.float)  # remove the first time_0 data that is affected by the impact force
                graph_sub.ndata['acc_Y'] = graph_sub.ndata['acc_Y'] / torch.max(torch.abs(graph_sub.ndata['acc_Y'])) # normalization
                graph_sub.ndata['phi_Y'] = torch.tensor(self.phi[graph_id][:, 0:self.mode_N], dtype = torch.float)
                graph_sub.ndata['node'] = torch.tensor(self.node[graph_id], dtype = torch.float)
                
                # feature propagation ##########################################
                # define node mask
                node_mask = torch.zeros(len(self.node[graph_id]), dtype=torch.bool)
                known_indices = np.array([1,4,5,7,9,12,15,18,21,23,26,29])-1 # only consider girder
                node_mask[known_indices] = True
                graph_sub.ndata['mask'] = node_mask
                # propagate features
                # missing_ratio = np.count_nonzero(node_mask == False)/len(node_mask)
                # print('missing_ratio =', missing_ratio)
                edge_index =  torch.LongTensor([list(row) for row in list(zip(src, dst))]).T
                acc_complete = graph_sub.ndata['acc_Y']
                acc_FP = Feature_Propagation(acc_complete, node_mask, edge_index, len(self.node[graph_id]))
                acc_FP[acc_complete[:, 1]==0, :] = 0 # reponses of constrained DOFs should be zero
                graph_sub.ndata['acc_Y'] = acc_FP

                # define graph features ##########################################
                graph_freq = self.freq[graph_id][:self.mode_N].squeeze()
                self.graphs.append(graph_sub)
                self.freqs.append(graph_freq)
                # print('graph_id =', graph_id)
        # Convert the graph features to tensor type
        self.freqs = torch.tensor(np.array(self.freqs), dtype = torch.float)
    def __getitem__(self, i):
        return self.graphs[i], self.freqs[i]
    def __len__(self):
        return len(self.graphs)

def get_dataset(data_path, graph_no, time_0_list, time_L, mode_N):
    # Load data
    mat_contents = sio.loadmat(data_path)
    acc_input = mat_contents['acceleration_time_out'][:, 0]
    freq = mat_contents['frequency_out'][:, 0]
    phi = mat_contents['modeshape_out'][:, 0]   # true mode shape
    node = mat_contents['node_out'][:, 0]
    element = mat_contents['element_out'][:, 0]
    dataset = Dataset(graph_no, time_0_list, time_L, mode_N, 
                 acc_input, phi, node, # node features
                 element, # edge features
                 freq)  # graph features
    return dataset