# import python lib
import torch
import dgl
from dgl.data import DGLDataset
import numpy as np
import scipy.io as sio # load .mat file

# Load data
skip_L = 2000 # skip the first skip_L data points because of the unwanted impact response
time_L = 2000 # length of input time series
modeN = 7  # number of modes to be identified
mat_contents = sio.loadmat("C:/Users/14360/Desktop/trapezoid_time_bottom_input_100sample_lowpass.mat")
acc_input = mat_contents['acceleration_time_out'][:, 0]
freq = mat_contents['frequency_out'][:, 0]
phi = mat_contents['modeshape_out'][:, 0]   # true mode shape
# phi = abs(mat_contents['modeshape_out'][:, 0])*1  # absolute mode shape
damping = mat_contents['damping_out'][:, 0]
node = mat_contents['node_out'][:, 0]
element = mat_contents['element_out'][:, 0]
element_L = mat_contents['element_L_out'][:, 0]
element_theta = mat_contents['element_theta_out'][:, 0]

# Use GPU for training
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", device)

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
            label = 0
            self.graphs.append(graph_sub.to(device))
            self.labels.append(label)
            print('graph_id =', graph_id)
        # Convert the label list to tensor for saving.
        self.labels = torch.LongTensor(self.labels)
    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]
    def __len__(self):
        return len(self.graphs)


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

bs = 32
dataloader_train = dgl.dataloading.GraphDataLoader(train_set, batch_size=bs,
                              drop_last=False, shuffle=True)
dataloader_valid = dgl.dataloading.GraphDataLoader(valid_set, batch_size=bs,
                              drop_last=False, shuffle=True)
print('Dataloader: done')