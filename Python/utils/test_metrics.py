import torch
import numpy as np

from scipy.signal import welch

from utils.match_mode import match_mode
from utils.damping_identification import RDT
import yaml

# Load config
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)


def test_metrics_single(graph_test, mode_id_N, model_test):              
    model = model_test.to('cpu') # only test on CPU
    with torch.no_grad(): # Turn off gradient tracking
        q_pred_test, phi_pred_test = model(graph_test[0].to('cpu'))

    q_pred_test = torch.squeeze(q_pred_test, 0)
    
    # calulate PSD using Scipy ########
    frequencies, q_pred_test_fft = welch(q_pred_test.T.numpy(),
                             fs=config['shared']['fs'], nperseg=256, nfft=2048)
    q_pred_test_fft = torch.from_numpy(q_pred_test_fft)
    # sort the results #######
    _, q_pred_test_fft_max_indices = torch.max(q_pred_test_fft.T, dim=0)
    q_pred_test_sorted_indices = torch.argsort(q_pred_test_fft_max_indices)
    q_pred_test = q_pred_test[:, q_pred_test_sorted_indices] # sort q
    phi_pred_test = phi_pred_test[:, q_pred_test_sorted_indices] # sort phi
    q_pred_test_fft = q_pred_test_fft[q_pred_test_sorted_indices, :] # sort FFT.abs of q's autocorrelation
    
    # frequency identification - sort the frequency again #######################
    q_pred_test_fft_max, q_pred_test_fft_max_indices = torch.max(q_pred_test_fft.T, dim=0) 
    freq_pred = frequencies[q_pred_test_fft_max_indices]
    freq_true = graph_test[1].squeeze().numpy()
    
    # MAC identification - sort the frequency again #######################
    # use all DOFs to calculate MAC
    phi_true = graph_test[0].ndata['phi_Y'].numpy().squeeze()
    phi_pred = phi_pred_test.numpy().squeeze()
    # only use known DOF to calculate MAC (use this only for laboratory test)
    # phi_true = phi_true[graph_test[0].ndata['mask'], :]
    # phi_pred = phi_pred[graph_test[0].ndata['mask'], :]

    # Match identified modes with true modes based on their frequencies
    freq_pred_match, MAC_pred_match, match_i = match_mode(freq_true, freq_pred, phi_true, phi_pred)
    freq_id = np.array(freq_pred_match)
    phi_id_MAC = np.array(MAC_pred_match)
    
    # damping identification
    zeta_id = np.zeros(len(match_i))
    for i in range(len(match_i)):
        zeta_id[i] = RDT(q_pred_test[:, match_i[i]].numpy(), 
                              fs=config['shared']['fs'], 
                              threshold=None, 
                              segment_length_sec=5)
        zeta_id[i] = zeta_id[i] * 100 # damping ratio in %

    return freq_id, phi_id_MAC, zeta_id