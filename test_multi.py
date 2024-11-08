# pytorch lib
import torch
import numpy as np
import time
import matplotlib.pyplot as plt

from scipy.signal import welch

from models.models import create_model
from data.dataset_loader import get_dataset
from utils.match_mode import MAC
from utils.match_mode import match_mode
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
# %% plot test results
N_test = len(test_no)
plt.close('all')
freq_test_true = np.zeros([N_test, 5])
freq_test_id = np.zeros([N_test, 5])
MAC_test_id = np.zeros([N_test, 5])
count = 0
start_time = time.time()
for graph_test in dataloader_test:
    
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
    
    # record identified frequencies
    # sort the frequency again
    q_pred_test_fft_max, q_pred_test_fft_max_indices = torch.max(q_pred_test_fft.T, dim=0) 
    freq_pred = frequencies[q_pred_test_fft_max_indices]
    # record true frequencies
    freq_true = graph_test[1].squeeze().numpy()
    
    phi_true = graph_test[0].ndata['phi_Y'].detach().numpy().squeeze()
    phi_pred = phi_pred_test.detach().numpy().squeeze()

    freq_pred_match, MAC_pred_match = match_mode(freq_true, freq_pred, phi_true, phi_pred)
    freq_test_true[count, :] = freq_true[:5]
    freq_test_id[count, :] = np.array(freq_pred_match)
    MAC_test_id[count, :] = np.array(MAC_pred_match)
    
    print(count)
    count = count + 1

test_time = (time.time() - start_time)
print("Test the model: done, %s seconds" % test_time)   

# %% visulize results
marker_size = 15
plt.close('all')

# frequency results
plt.figure(figsize=(4*2.54/2.54, 4*2.54/2.54))

plt.scatter(freq_test_true[:, 0], freq_test_id[:, 0], color='#FF1F5B', s=marker_size, alpha=1, label='mode 1')
plt.scatter(freq_test_true[:, 1], freq_test_id[:, 1], color='#00CD6C', s=marker_size, alpha=1, label='mode 2')
plt.scatter(freq_test_true[:, 2], freq_test_id[:, 2], color='#009ADE', s=marker_size, alpha=1, label='mode 3')
plt.scatter(freq_test_true[:, 3], freq_test_id[:, 3], color='#AF58BA', s=marker_size, alpha=1, label='mode 4')
plt.scatter(freq_test_true[:, 4], freq_test_id[:, 4], color='#FFC61E', s=marker_size, alpha=1, label='mode 5')
    
plt.plot([0,20], [0,20], linestyle='--', color='black', label='\u00B1 0%')
plt.plot([0,20], [0,20*0.9], linestyle='--', color='blue', label='\u00B1 10%')
plt.plot([0,20*0.9], [0,20], linestyle='--', color='blue')
plt.xlim([-0.05,20])
plt.ylim([-0.05,20])
plt.legend(prop={'family': 'Times New Roman', 'size': 16}, handlelength=1, borderpad=0.1, labelspacing=0.1)
plt.xticks(fontsize=15, fontname='Times New Roman')
plt.yticks(fontsize=15, fontname='Times New Roman')
plt.xlabel('True Frequency (Hz)', fontname='Times New Roman', fontsize=15)
plt.ylabel('Identified Frequency (Hz)', fontname='Times New Roman', fontsize=15)
plt.grid()
plt.tight_layout()
plt.show()

# mode shape results
# mode shape MAC results of testing set
plt.figure(figsize=(4*2.54/2.54, 4*2.54/2.54))
plt.scatter(np.zeros([N_test, 1])+1, MAC_test_id[:, 0], color='#FF1F5B', label='mode 1', s=marker_size, alpha=1)
plt.scatter(np.zeros([N_test, 1])+2, MAC_test_id[:, 1], color='#00CD6C', label='mode 2', s=marker_size, alpha=1)
plt.scatter(np.zeros([N_test, 1])+3, MAC_test_id[:, 2], color='#009ADE', label='mode 3', s=marker_size, alpha=1)
plt.scatter(np.zeros([N_test, 1])+4, MAC_test_id[:, 3], color='#AF58BA', label='mode 4', s=marker_size, alpha=1)
plt.scatter(np.zeros([N_test, 1])+5, MAC_test_id[:, 4], color='#FFC61E', label='mode 5', s=marker_size, alpha=1)
plt.boxplot(MAC_test_id, 0, '')
plt.xticks([1, 2, 3, 4, 5], ['Mode1', 'Mode2', 'Mode3', 'Mode4', 'Mode5'])
plt.ylim([0, 1.05])
plt.xticks(fontsize=15, fontname='Times New Roman')
plt.yticks(fontsize=15, fontname='Times New Roman')
plt.ylabel('MAC', fontname='Times New Roman', fontsize=17)
plt.grid()
plt.tight_layout()
plt.show()