# pytorch lib
import torch
import numpy as np
import time
import matplotlib.pyplot as plt

from scipy.signal import welch

from models.models import create_model
from data.dataset_loader import get_dataset
# from utils.autocorrelation import autocorrelation
import yaml

# Load config
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

model = create_model(model_name=config['model']['name'],
                      dim=config['model']['dim'], 
                      time_L=config['shared']['time_L'], 
                      fft_n=config['model']['fft_n'], 
                      mode_N=config['shared']['mode_N'], 
                      dropout_rate=config['model']['dropout_rate'], 
                      hid_layer=config['model']['hid_layer'])

PATH = config['model']['name'] + ".pth"
model.load_state_dict(torch.load(PATH))

model.eval()

# designate sample no. for testing ######################################
test_no = np.array(range(32)) # sample from the training set
# test_no = np.array(range(40, 50)) # sample from the testing set
dataloader_test = get_dataset(data_path="C:/Users/14360/Desktop/truss_500_lowpass.mat", 
                        bs=1, 
                        graph_no=test_no, 
                        time_0=config['shared']['time_0'], 
                        time_L=config['shared']['time_L'], 
                        mode_N=config['shared']['mode_N'],
                        device='cpu')
print('Create dataset: done')

# define a function to calculate MAC
def MAC(x, y):
    assert x.shape == y.shape, "Mode shapes must have the same shape"
    numerator = np.abs(np.dot(x, y.T))**2
    denominator_x = np.dot(x, x.T)
    denominator_y = np.dot(y, y.T)
    mac = numerator / (denominator_x * denominator_y)
    return mac
# %% plot test results
plt.close('all')
freq_pairs = []
phi1_MACs, phi2_MACs, phi3_MACs, phi4_MACs, phi5_MACs, phi6_MACs = [], [], [], [], [], []
count = 0
start_time = time.time()
for graph_test in dataloader_test:
    print(count)
    count = count + 1
    # model inference
    node_mask = graph_test[0].ndata['mask']
    node_test = graph_test[0].ndata['node']
    
    freq_test = graph_test[1].squeeze()
    damping_test = graph_test[2].squeeze()
    
    element_test = torch.stack((graph_test[0].edges()[0], graph_test[0].edges()[1]), dim=1)
    element_test = element_test[:element_test.size(0)//2, :]  # only select the first half because the graph is undirectional
    
    q_pred_test, phi_pred_test = model(graph_test[0])
    q_pred_test = torch.squeeze(q_pred_test, 0)
    
    # calulate PSD using Scipy ########
    frequencies, q_pred_test_fft = welch(q_pred_test.T.to('cpu').detach().numpy(),
                             fs=200, nperseg=256, nfft=config['model']['fft_n'])
    q_pred_test_fft = torch.from_numpy(q_pred_test_fft)
    # sort the results #######
    _, q_pred_test_fft_max_indices = torch.max(q_pred_test_fft.T, dim=0)
    q_pred_test_sorted_indices = torch.argsort(q_pred_test_fft_max_indices)
    q_pred_test = q_pred_test[:, q_pred_test_sorted_indices] # sort q
    phi_pred_test = phi_pred_test[:, q_pred_test_sorted_indices] # sort phi
    q_pred_test_fft = q_pred_test_fft[q_pred_test_sorted_indices, :] # sort FFT.abs of q's autocorrelation
    
    acc_pred = phi_pred_test @ q_pred_test.T
    acc_true = graph_test[0].ndata['acc_Y']
    
    # record identified frequencies ##########################################
    q_pred_test_fft_max, q_pred_test_fft_max_indices = torch.max(q_pred_test_fft.T, dim=0) # sort the frequency again
    # # Combine tensor1 and tensor2 into one tensor for processing
    # combined = torch.stack((q_pred_test_fft_max, q_pred_test_fft_max_indices), dim=1)
    # # Sort by tensor2 and then by tensor1
    # _, unique_indices = torch.unique(q_pred_test_fft_max_indices, return_inverse=True)
    # # For each unique element in tensor2, keep the one with the maximum tensor1
    # keep_indices = []
    # for i in torch.unique(unique_indices):
    #     indices = (unique_indices == i).nonzero(as_tuple=True)[0]
    #     max_idx = indices[torch.argmax(q_pred_test_fft_max[indices])]
    #     keep_indices.append(max_idx)
    # # Get the filtered tensors
    # filtered_q_pred_test_fft_max_indices = q_pred_test_fft_max_indices[np.array(keep_indices)]
    freq_pred = frequencies[q_pred_test_fft_max_indices]
    # record true frequencies
    freq_true = graph_test[1].squeeze().numpy()
    # Define a similarity threshold
    tolerance = 0.5
    # List to store unique pairs along with their indices
    similar_pairs = []
    pair_indices = []
    # Find all unique pairs (a, b) where |a - b| <= tolerance, along with their indices
    for i, a in enumerate(freq_true):
        for j, b in enumerate(freq_pred):
            if abs(a - b) <= tolerance:
                # Store the pair in a consistent order (smallest first) to avoid duplicates
                pair = (min(a, b), max(a, b))
                indices = (i, j)
    
                # Check if the pair already exists to ensure uniqueness
                if pair not in similar_pairs:
                    similar_pairs.append(pair)
                    pair_indices.append(indices)
    freq_pairs.append(similar_pairs)
    
    # record identified mode shapes ##########################################
    phi_pred = phi_pred_test.detach().numpy().squeeze()
    phi_true = graph_test[0].ndata['phi_Y'].detach().numpy().squeeze()
    for i in range(len(pair_indices)):     
        pair_indice = np.array(pair_indices[i])
        if pair_indice[0] == 0: # first column is the mode no. of true mode shapes
            phi1_MACs.append(MAC(phi_true[:, pair_indice[0]], phi_pred[:, pair_indice[1]]))
        elif pair_indice[0] == 1:
            phi2_MACs.append(MAC(phi_true[:, pair_indice[0]], phi_pred[:, pair_indice[1]]))
        elif pair_indice[0] == 2:
            phi3_MACs.append(MAC(phi_true[:, pair_indice[0]], phi_pred[:, pair_indice[1]]))
        elif pair_indice[0] == 3:
            phi4_MACs.append(MAC(phi_true[:, pair_indice[0]], phi_pred[:, pair_indice[1]]))
        elif pair_indice[0] == 4:
            phi5_MACs.append(MAC(phi_true[:, pair_indice[0]], phi_pred[:, pair_indice[1]]))
        elif pair_indice[0] == 5:
            phi6_MACs.append(MAC(phi_true[:, pair_indice[0]], phi_pred[:, pair_indice[1]]))

test_time = (time.time() - start_time)
print("Train the model: done, %s seconds" % test_time)   

# %% visulize results
# frequency results of training set
plt.figure(figsize=(4*2.54/2.54, 4*2.54/2.54))
for freq_pair in freq_pairs:
    freq_pair = np.array(freq_pair)
    plt.scatter(freq_pair[:, 0], freq_pair[:, 1], color='#FF1F5B', s=12, alpha=1)
plt.plot([0,20], [0,20], linestyle='--', color='black', label='\u00B1 0%')
plt.plot([0,20], [0,20*0.9], linestyle='--', color='blue', label='\u00B1 10%')
plt.plot([0,20*0.9], [0,20], linestyle='--', color='blue')
plt.xlim([0,25])
plt.ylim([0,25])
plt.legend(prop={'family': 'Times New Roman', 'size': 16}, handlelength=1, borderpad=0.1, labelspacing=0.1)
plt.xticks(fontsize=18, fontname='Times New Roman')
plt.yticks(fontsize=18, fontname='Times New Roman')
plt.xlabel('True Frequency (Hz)', fontname='Times New Roman', fontsize=18)
plt.ylabel('Identified Frequency (Hz)', fontname='Times New Roman', fontsize=18)
plt.grid()
plt.tight_layout()
plt.show()