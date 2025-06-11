# pytorch lib
import torch
import numpy as np
import math
import dgl
import matplotlib.pyplot as plt

from scipy.signal import welch

from models.models import create_model
from data.dataset_loader_laboratory import get_dataset
from utils.match_mode import match_mode
from utils.damping_identification import RDT
import yaml

from torchinfo import summary

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load config
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

model = create_model(model_name=config['model']['name'],
                      dim=config['model']['dim'], 
                      time_L=config['shared']['time_L'], 
                      mode_N=config['shared']['mode_N'], 
                      dropout_rate=config['model']['dropout_rate'], 
                      hid_layer=config['model']['hid_layer'])

PATH = config['model']['name'] + ".pth"
model.load_state_dict(torch.load(PATH))

# output the information of the model
# print(summary(model))

model.eval()

# designate sample no. for testing ######################################
test_no = np.array([4-1]) 
dataset_test = get_dataset(data_path=config['data']['path'], 
                        graph_no=test_no, 
                        time_0_list=config['shared']['time_0'], 
                        time_L=config['shared']['time_L'], 
                        mode_N=config['shared']['mode_N'])
dataloader_test = dgl.dataloading.GraphDataLoader(dataset_test, 
                              batch_size=1,
                              drop_last=False, shuffle=False)
print('Create dataset: done')
# %% plot figures for the paper

fontsize = 14

plt.close('all')
count = 0
for graph_test in dataloader_test:
# graph_test = test_data[0].to('cpu')
    node_mask = graph_test[0].ndata['mask']
    node_test = graph_test[0].ndata['node']
    
    freq_test = graph_test[1].squeeze()
    
    element_test = torch.stack((graph_test[0].edges()[0], graph_test[0].edges()[1]), dim=1)
    element_test = element_test[:element_test.size(0)//2, :]  # only select the first half because the graph is undirectional
    
    with torch.no_grad(): # Turn off gradient tracking
        q_pred_test, phi_pred_test = model(graph_test[0])
    q_pred_test = torch.squeeze(q_pred_test, 0)
    
    # sort out q from low-order modes to high-order modes, based on the dominant frequency, then also sort out phi
    # calulate PSD using Scipy ########
    frequencies, q_pred_test_fft = welch(q_pred_test.T.to('cpu').detach().numpy(),
                             fs=config['shared']['fs'], nperseg=512, nfft=config['model']['fft_n'])
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
    q_pred_test_fft_max, q_pred_test_fft_max_indices = torch.max(q_pred_test_fft.T, dim=0) # sort the frequency again
    freq_pred = frequencies[q_pred_test_fft_max_indices]
    # record true frequencies
    freq_true = freq_test.numpy()
    
    
    # Set font to Times New Roman
    plt.rcParams['font.family'] = 'Times New Roman'
    
    # plot dynamic measurements ########################################
    fig, ax = plt.subplots(6, 2, figsize=(15 * 0.3937, 12 * 0.3937), layout="constrained")
    # Set column titles
    ax[0, 0].set_title(r'Acceleration ($m/s^2$)', fontsize=fontsize)
    ax[0, 1].set_title(r'PSD of Acceleration', fontsize=fontsize)
    
    DOFs = np.array([4, 8, 11, 14, 17, 20])
    i = 0
    while i < 6:
        # Plot acc
        ax[i, 0].plot(acc_true[DOFs[i], :].numpy())
        ax[i, 0].grid()
    
        # Plot PSD of acc
        frequencies, psd = welch(acc_true[DOFs[i], :],
                                 fs=config['shared']['fs'], nperseg=512, nfft=config['model']['fft_n'])
        ax[i, 1].plot(frequencies, psd, label=r'$\mathrm{A}_{%d}$' % (i+1), linewidth=1)
        ax[i, 1].grid(True)
        ax[i, 1].set_xlim(0, 20)
        ax[i, 1].legend(fontsize=fontsize*0.6, loc='upper right')
        
        # Only set x-axis labels for the last row
        if i == 6 - 1:
            ax[i, 0].set_xlabel('Time Step', fontsize=fontsize)
            ax[i, 1].set_xlabel('Frequency (Hz)', fontsize=fontsize)
        else:
            ax[i, 0].set_xticklabels([])
            ax[i, 1].set_xticklabels([])
        
        i = i + 1
    
    # plot modal decomposition results ########################################
    fig, ax = plt.subplots(config['shared']['mode_N'], 3,
                           figsize=(30 * 0.3937, 11 * 0.3937), layout="constrained")
    
    # Set the figure title
    # title_text = "Truss No.={:.0f}".format(test_no[count] + 1)
    # fig.suptitle(title_text, fontsize=16)
    
    # Set column titles
    ax[0, 0].set_title(r'Modal response $Q(t)$', fontsize=fontsize)
    ax[0, 1].set_title(r'PSD of $Q(t)$', fontsize=fontsize)
    ax[0, 2].set_title(r'Mode shape', fontsize=fontsize)
    
    for mode_no in range(config['shared']['mode_N']):
        # Plot modal responses
        ax[mode_no, 0].plot(q_pred_test[:, mode_no].to('cpu').detach().numpy())
        ax[mode_no, 0].grid()
    
        # Plot PSD of modal responses
        psd = q_pred_test_fft[mode_no, :].to('cpu').detach().numpy()
        ax[mode_no, 1].plot(frequencies, psd)
        ax[mode_no, 1].grid(True)
        ax[mode_no, 1].set_xlim(0, 20)
    
        # Plot vertical lines at test frequencies
        for i in range(config['shared']['mode_N']):
            ax[mode_no, 1].plot([freq_test[i], freq_test[i]], [0, max(psd)], color='#FF1F5B',linestyle='--')
    
        # Plot mode shapes
        phi_pred = phi_pred_test[:, mode_no].to('cpu').detach().numpy().squeeze()
        phi_pred = phi_pred / max(abs(phi_pred)) * 1  # max normalization
        node_pred = np.zeros([len(node_test), 2])
        node_pred[:, 0] = node_test[:, 0]
        node_pred[:, 1] = node_test[:, 1] + phi_pred
    
        phi_true = graph_test[0].ndata['phi_Y'][:, mode_no].detach().numpy().squeeze() * 1
        node_true = np.zeros([len(node_test), 2])
        node_true[:, 0] = node_test[:, 0]
        node_true[:, 1] = node_test[:, 1] + phi_true
    
        for ele in element_test:
            node1 = node_test[ele[0]]
            node2 = node_test[ele[1]]
            ax[mode_no, 2].plot([node1[0], node2[0]], [node1[1], node2[1]], 'k--', linewidth=0.75)
    
        for ele in element_test:
            node1 = node_pred[ele[0]]
            node2 = node_pred[ele[1]]
            ax[mode_no, 2].plot([node1[0], node2[0]], [node1[1], node2[1]], 'b-')
    
        ax[mode_no, 2].plot(node_pred[:, 0], node_pred[:, 1], 'o', markersize=3, label='known', color='#FF1F5B')
        ax[mode_no, 2].plot(node_pred[~node_mask, 0], node_pred[~node_mask, 1], 's', markersize=3, label='unknown', color='#00CD6C')
        ax[mode_no, 2].grid()
        ax[mode_no, 2].set_ylim(-1.1,1.1)
    
        # Only set x-axis labels for the last row
        if mode_no == config['shared']['mode_N'] - 1:
            ax[mode_no, 0].set_xlabel('Time Step', fontsize=fontsize)
            ax[mode_no, 1].set_xlabel('Frequency (Hz)', fontsize=fontsize)
            ax[mode_no, 2].set_xlabel('X (m)', fontsize=fontsize)
        else:
            ax[mode_no, 0].set_xticklabels([])
            ax[mode_no, 1].set_xticklabels([])
            ax[mode_no, 2].set_xticklabels([])
    
        count += 1

phi_pred_export = np.array(phi_pred_test)
