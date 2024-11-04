# pytorch lib
import torch
import numpy as np
import math
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
# test_no = np.array([4-1]) # sample from the training set, 24 is for APWSHM paper
test_no = np.array([97-1]) # sample from the testing set, 47 is for APWSHM paper
dataloader_test = get_dataset(data_path="C:/Users/14360/Desktop/truss_500_lowpass.mat", 
                        bs=config['data']['bs'], 
                        graph_no=test_no, 
                        time_0=config['shared']['time_0'], 
                        time_L=config['shared']['time_L'], 
                        mode_N=config['shared']['mode_N'],
                        device='cpu')
print('Create dataset: done')

def MAC(x, y):
    assert x.shape == y.shape, "Mode shapes must have the same shape"
    numerator = np.abs(np.dot(x, y.T))**2
    denominator_x = np.dot(x, x.T)
    denominator_y = np.dot(y, y.T)
    mac = numerator / (denominator_x * denominator_y)
    return mac
# %% plot test results
# plt.close('all')
# count = 0
# for graph_test in dataloader_test:
# # graph_test = test_data[0].to('cpu')
#     node_mask = graph_test[0].ndata['mask']
#     node_test = graph_test[0].ndata['node']
    
#     freq_test = graph_test[1].squeeze()
#     damping_test = graph_test[2].squeeze()
    
#     element_test = torch.stack((graph_test[0].edges()[0], graph_test[0].edges()[1]), dim=1)
#     element_test = element_test[:element_test.size(0)//2, :]  # only select the first half because the graph is undirectional
    
#     q_pred_test, phi_pred_test = model(graph_test[0])
#     # phi_pred_test = graph_test.ndata['phi_Y']  # use true phi for testing
#     q_pred_test = torch.squeeze(q_pred_test, 0)
    
#     # sort out q from low-order modes to high-order modes, based on the dominant frequency, then also sort out phi
#     # calulate PSD using FFT of q's autocorrelation#########
#     # q_pred_test_auto = torch.zeros_like(q_pred_test.T)
#     # for i in range(0, q_pred_test_auto.size(0)):
#     #     q_pred_test_auto[i, :] = autocorrelation(q_pred_test.T[i, :])
#     # q_pred_test_fft = torch.fft.rfft(q_pred_test.T, n=config['model']['fft_n']).abs()  # FFT of q
#     # q_pred_test_fft = torch.fft.rfft(q_pred_test_auto, n=config['model']['fft_n']).abs()  # FFT of q's autocorrelation
#     # plt.figure()
#     # plt.plot(q_pred_test_auto[2, :].detach().numpy())
#     # calulate PSD using Scipy ########
#     frequencies, q_pred_test_fft = welch(q_pred_test.T.to('cpu').detach().numpy(),
#                              fs=200, nperseg=256, nfft=config['model']['fft_n'])
#     q_pred_test_fft = torch.from_numpy(q_pred_test_fft)
#     # sort the results #######
#     _, q_pred_test_fft_max_indices = torch.max(q_pred_test_fft.T, dim=0)
#     q_pred_test_sorted_indices = torch.argsort(q_pred_test_fft_max_indices)
#     q_pred_test = q_pred_test[:, q_pred_test_sorted_indices] # sort q
#     phi_pred_test = phi_pred_test[:, q_pred_test_sorted_indices] # sort phi
#     q_pred_test_fft = q_pred_test_fft[q_pred_test_sorted_indices, :] # sort FFT.abs of q's autocorrelation
    
    
    
    
#     acc_pred = phi_pred_test @ q_pred_test.T
#     acc_true = graph_test[0].ndata['acc_Y']

#     # plot acc time series and PSD #########################################
#     dof_no = np.array([20, 25, 30, 35, 40]) - 1  
#     fig, ax = plt.subplots(5, 2, layout="constrained")
#     title_text = "Truss No.={:.0f}".format(test_no[count]+1)
#     fig.suptitle(title_text, fontsize=16)
#     for i in range(5):
#         ax[i, 0].plot(acc_true[dof_no[i], :].detach().numpy(), linestyle='--', label='True')
#         ax[i, 0].plot(acc_pred[dof_no[i], :].detach().numpy(), label='Pred')
#         ax[i, 0].set_ylabel('Acc', fontsize=14)
#         ax[i, 0].grid()
#         title_text = "DOF={:.0f}".format(dof_no[i]+1)
#         ax[i, 0].set_title(title_text, fontsize=14)
#         ax[i, 0].legend()
        
#         frequencies, psd_pred = welch(acc_pred[dof_no[i], :].to('cpu').detach().numpy(),
#                                  fs=200, nperseg=256, nfft=config['model']['fft_n'])
#         frequencies, psd_true = welch(acc_true[dof_no[i], :].to('cpu').detach().numpy(),
#                                  fs=200, nperseg=256, nfft=config['model']['fft_n'])
#         ax[i, 1].plot(frequencies, psd_true, linestyle='--', label='True')
#         ax[i, 1].plot(frequencies, psd_pred, label='Pred')
#         ax[i, 1].set_xlabel('Frequency [Hz]', fontsize=14)
#         ax[i, 1].set_ylabel('PSD', fontsize=14)
#         ax[i, 1].grid(True)
#         ax[i, 1].legend()
#         ax[i, 1].set_xlim(0, 50)
#         for j in range(5):
#             ax[i, 1].plot([freq_test[j], freq_test[j]], [0, max(psd_pred)], color='#FF1F5B')   
    
#     # plot PSD of modal responses #########################################
#     fig, ax = plt.subplots(config['shared']['mode_N'], 4, layout="constrained")
#     title_text = "Truss No.={:.0f}".format(test_no[count]+1)
#     fig.suptitle(title_text, fontsize=16)
#     for mode_no in range(config['shared']['mode_N']):
#         ax[mode_no, 0].plot(q_pred_test[:, mode_no].to('cpu').detach().numpy())
#         ax[mode_no, 0].set_ylabel('Modal acc')
#         ax[mode_no, 0].grid()
        
#         # [f_n, zeta] = ModalId(frequencies, psd)
#         psd = q_pred_test_fft[mode_no, :].to('cpu').detach().numpy()
#         ax[mode_no, 1].plot(frequencies, psd)
#         # title_text = "component={:.0f}, f={:.3f}, zeta={:.5}".format(mode_no+1, f_n, zeta)
#         # ax[mode_no, 1].set_title(title_text)
#         ax[mode_no, 1].set_xlabel('Frequency [Hz]')
#         ax[mode_no, 1].set_ylabel('PSD')
#         ax[mode_no, 1].grid(True)
#         ax[mode_no, 1].set_xlim(0, 50)
#         for i in range(7):
#             ax[mode_no, 1].plot([freq_test[i], freq_test[i]], [0, max(psd)], color='#FF1F5B')    
        
#         phi_pred = phi_pred_test[:, mode_no].to('cpu').detach().numpy().squeeze()
#         phi_pred = phi_pred / max(abs(phi_pred)) * 2
#         node_pred = np.zeros([len(node_test), 2])
#         node_pred[:, 0] = node_test[:, 0]
#         node_pred[:, 1] = node_test[:, 1] + phi_pred
        
#         phi_true = graph_test[0].ndata['phi_Y'][:, mode_no].detach().numpy().squeeze() * 2
#         node_true = np.zeros([len(node_test), 2])
#         node_true[:, 0] = node_test[:, 0]
#         node_true[:, 1] = node_test[:, 1] + phi_true
        
#         # show the complexity of mode shapes - curvature
#         phi_true_bottom = phi_true[node_test[:, 1] == 0]
#         phi_true_bottom = np.concatenate((phi_true_bottom[1:], phi_true_bottom[:1]))
#         # print( np.sqrt(np.sum(np.diff(phi_true_bottom)**2)) )
#         phi_pred_bottom = phi_pred[node_test[:, 1] == 0]
#         phi_pred_bottom = np.concatenate((phi_pred_bottom[1:], phi_pred_bottom[:1]))
#         # print(np.sqrt(np.sum(np.diff(phi_pred_bottom)**2)) )
        
        
#         for ele in element_test:
#             node1 = node_test[ele[0]]
#             node2 = node_test[ele[1]]
#             ax[mode_no, 3-1].plot([node1[0], node2[0]], [node1[1], node2[1]], 'k--')
#         # ax[mode_no, 2].plot(node_true[:, 0], node_true[:, 1], 'ko', markersize=7, label='undeformed')        
#         for ele in element_test:
#             node1 = node_pred[ele[0]]
#             node2 = node_pred[ele[1]]
#             ax[mode_no, 3-1].plot([node1[0], node2[0]], [node1[1], node2[1]], 'b-')
            
#         # ax[mode_no, 2].plot(node_pred[:, 0], node_pred[:, 1], 'bo', markersize=7, label='predicted')
        
#         ax[mode_no, 3-1].plot(node_pred[:, 0], node_pred[:, 1], 'o', markersize=3, label='identified_known', color='#AF58BA')
#         ax[mode_no, 3-1].plot(node_pred[~node_mask, 0], node_pred[~node_mask, 1], 's', markersize=3, label='identified_unknown', color='#00CD6C')
#         ax[mode_no, 3-1].grid()
#         ax[mode_no, 3-1].set_ylabel('shape_id')
#         title_text = "curvature={:.3f}".format(np.sqrt(np.sum(np.diff(phi_pred_bottom)**2)))
#         ax[mode_no, 3-1].set_title(title_text, fontsize=12)  
        
#         # ax[mode_no, 2].hist(q_pred_test[:, mode_no].to('cpu').detach().numpy(), bins=100, edgecolor='black')
#         # ax[mode_no, 2].set_title('Histogram of Modal Responses')
#         # ax[mode_no, 2].set_xlabel('Value')
#         # ax[mode_no, 2].set_ylabel('Frequency')
#         # ax[mode_no, 2].grid(True)
        
#         for ele in element_test:
#             node1 = node_test[ele[0]]
#             node2 = node_test[ele[1]]
#             ax[mode_no, 4-1].plot([node1[0], node2[0]], [node1[1], node2[1]], 'k--')
#         # ax[mode_no, 3].plot(node_true[:, 0], node_true[:, 1], 'ko', markersize=7, label='undeformed')    
#         for ele in element_test:
#             node1 = node_true[ele[0]]
#             node2 = node_true[ele[1]]
#             ax[mode_no, 4-1].plot([node1[0], node2[0]], [node1[1], node2[1]], 'r-')
#         ax[mode_no, 4-1].plot(node_true[:, 0], node_true[:, 1], 'ro', markersize=3, label='true')
#         # ax[mode_no, 2].set_aspect('equal')
#         ax[mode_no, 4-1].grid()
#         # ylabel_text = "shape={:.0f}".format(mode_no+1)
#         ax[mode_no, 4-1].set_ylabel('shape_true')
#         title_text = "mode={:.0f}, f={:.3f}, zeta={:.4f}, curvature={:.3f}".format(mode_no+1, freq_test[mode_no], damping_test[mode_no], np.sqrt(np.sum(np.diff(phi_true_bottom)**2)))
#         ax[mode_no, 4-1].set_title(title_text, fontsize=12)  
        
#         count = count + 1 
       
# %% plot for APWSHM
plt.close('all')
count = 0
for graph_test in dataloader_test:
# graph_test = test_data[0].to('cpu')
    node_mask = graph_test[0].ndata['mask']
    node_test = graph_test[0].ndata['node']
    
    freq_test = graph_test[1].squeeze()
    damping_test = graph_test[2].squeeze()
    
    element_test = torch.stack((graph_test[0].edges()[0], graph_test[0].edges()[1]), dim=1)
    element_test = element_test[:element_test.size(0)//2, :]  # only select the first half because the graph is undirectional
    
    q_pred_test, phi_pred_test = model(graph_test[0])
    q_pred_test = torch.squeeze(q_pred_test, 0)
    
    # sort out q from low-order modes to high-order modes, based on the dominant frequency, then also sort out phi
    # calulate PSD using Scipy ########
    frequencies, q_pred_test_fft = welch(q_pred_test.T.to('cpu').detach().numpy(),
                             fs=200, nperseg=256, nfft=config['model']['fft_n'])
    q_pred_test_fft = torch.from_numpy(q_pred_test_fft)
    # sort the results #######
    _, q_pred_test_fft_max_indices = torch.max(q_pred_test_fft.T, dim=0)
    q_pred_test_sorted_indices = torch.argsort(q_pred_test_fft_max_indices)
    q_pred_test = q_pred_test[:, q_pred_test_sorted_indices] # sort q
    phi_pred_test = phi_pred_test[:, q_pred_test_sorted_indices] # sort phi
    # phi_pred_test = phi_pred_test[:, torch.tensor([4,0,2,1,5,6,3])] # sort phi - for graph_id 46
    q_pred_test_fft = q_pred_test_fft[q_pred_test_sorted_indices, :] # sort FFT.abs of q's autocorrelation
    
    acc_pred = phi_pred_test @ q_pred_test.T
    acc_true = graph_test[0].ndata['acc_Y']
    
    # record identified frequencies
    q_pred_test_fft_max, q_pred_test_fft_max_indices = torch.max(q_pred_test_fft.T, dim=0) # sort the frequency again
    freq_pred = frequencies[q_pred_test_fft_max_indices]
    # record true frequencies
    freq_true = graph_test[1].squeeze().numpy()
    
    # calculate MAC
    phi_pred = phi_pred_test.detach().numpy()
    phi_true = graph_test[0].ndata['phi_Y'].detach().numpy()
    MAC(phi_true[:,3], phi_pred[:,4])

    # plot acc time series and PSD #########################################
    dof_no = np.array([20, 25, 30, 35, 40]) - 1  
    fig, ax = plt.subplots(5, 2, layout="constrained")
    title_text = "Truss No.={:.0f}".format(test_no[count]+1)
    fig.suptitle(title_text, fontsize=16)
    for i in range(5):
        ax[i, 0].plot(acc_true[dof_no[i], :].detach().numpy(), linestyle='--', label='True')
        ax[i, 0].plot(acc_pred[dof_no[i], :].detach().numpy(), label='Pred')
        ax[i, 0].set_xlabel('Time Step', fontsize=14)
        ax[i, 0].set_ylabel('Acc', fontsize=14)
        ax[i, 0].grid()
        title_text = "DOF={:.0f}".format(dof_no[i]+1)
        ax[i, 0].set_title(title_text, fontsize=14)
        ax[i, 0].legend()
        
        frequencies, psd_pred = welch(acc_pred[dof_no[i], :].to('cpu').detach().numpy(),
                                 fs=200, nperseg=256, nfft=config['model']['fft_n'])
        frequencies, psd_true = welch(acc_true[dof_no[i], :].to('cpu').detach().numpy(),
                                 fs=200, nperseg=256, nfft=config['model']['fft_n'])
        ax[i, 1].plot(frequencies, psd_true, linestyle='--', label='True')
        ax[i, 1].plot(frequencies, psd_pred, label='Pred')
        ax[i, 1].set_xlabel('Frequency [Hz]', fontsize=14)
        ax[i, 1].set_ylabel('PSD', fontsize=14)
        ax[i, 1].grid(True)
        ax[i, 1].legend()
        ax[i, 1].set_xlim(0, 50)
        for j in range(5):
            ax[i, 1].plot([freq_test[j], freq_test[j]], [0, max(psd_pred)], color='#FF1F5B')   
    
    # plot PSD of modal responses #########################################
    fig, ax = plt.subplots(config['shared']['mode_N'], 4, layout="constrained")
    title_text = "Truss No.={:.0f}".format(test_no[count]+1)
    fig.suptitle(title_text, fontsize=16)
    for mode_no in range(config['shared']['mode_N']):
        ax[mode_no, 0].plot(q_pred_test[:, mode_no].to('cpu').detach().numpy())
        ax[mode_no, 0].set_ylabel('q')
        ax[mode_no, 0].set_xlabel('Time Step')
        ax[mode_no, 0].grid()    
        
        # [f_n, zeta] = ModalId(frequencies, psd)
        psd = q_pred_test_fft[mode_no, :].to('cpu').detach().numpy()
        ax[mode_no, 1].plot(frequencies, psd)
        ax[mode_no, 1].set_xlabel('Frequency [Hz]')
        ax[mode_no, 1].set_ylabel('PSD')
        ax[mode_no, 1].grid(True)
        ax[mode_no, 1].set_xlim(0, 50)
        for i in range(7):
            ax[mode_no, 1].plot([freq_test[i], freq_test[i]], [0, max(psd)], color='#FF1F5B')    
        
        phi_pred = phi_pred_test[:, mode_no].to('cpu').detach().numpy().squeeze()
        phi_pred = phi_pred / max(abs(phi_pred)) * 2
        node_pred = np.zeros([len(node_test), 2])
        node_pred[:, 0] = node_test[:, 0]
        node_pred[:, 1] = node_test[:, 1] + phi_pred
        
        phi_true = graph_test[0].ndata['phi_Y'][:, mode_no].detach().numpy().squeeze() * 2
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
        
        ax[mode_no, 3-1].plot(node_pred[:, 0], node_pred[:, 1], 'o', markersize=3, label='identified_known', color='#AF58BA')
        ax[mode_no, 3-1].plot(node_pred[~node_mask, 0], node_pred[~node_mask, 1], 's', markersize=3, label='identified_unknown', color='#00CD6C')
        ax[mode_no, 3-1].grid()
        ax[mode_no, 3-1].set_ylabel('phi_id')
        ax[mode_no, 3-1].set_xlabel('X [m]')
        
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
        ax[mode_no, 4-1].set_ylabel('phi_true')
        ax[mode_no, 4-1].set_xlabel('X [m]')
        title_text = "mode={:.0f}, f={:.3f} Hz".format(mode_no+1, freq_test[mode_no])
        ax[mode_no, 4-1].set_title(title_text, fontsize=12)  
        
        count = count + 1 

