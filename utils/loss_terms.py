import torch
import torch.nn as nn
import dgl
from utils.autocorrelation import autocorrelation
from scipy.signal import welch

# loss function using fft of modal responses
def loss_terms(q_pred, phi_pred, graph, fft_n):
    loss_func = nn.MSELoss()
    graph_unbatched = dgl.unbatch(graph[0])
    phi_index1 = 0
    phi_index2 = 0
    
    for i in range(len(graph_unbatched)):
        phi_index1 = phi_index2
        phi_index2 = phi_index1 + dgl.DGLGraph.number_of_nodes(graph_unbatched[i])
        phi_pred_unbatched = phi_pred[phi_index1:phi_index2]
        
        q_pred_unbatched = q_pred[i, :, :]
        # use FFT of q to calculate loss 3 ######################
        q_pred_unbatched_fft = torch.fft.rfft(q_pred_unbatched.T, n=fft_n).abs()
        # q_pred_unbatched_fft = q_pred_unbatched_fft / torch.max(q_pred_unbatched_fft)
        # use FFT of q's autocorrelation (PSD) to calculate loss 3 ######################
        # q_pred_auto = torch.zeros_like(q_pred_unbatched.T)
        # for mode_no in range(0, q_pred_auto.size(0)):
        #     q_pred_auto[mode_no, :] = autocorrelation(q_pred_unbatched.T[mode_no, :])
        # q_pred_unbatched_fft = torch.fft.rfft(q_pred_auto, n=fft_n).abs()  # FFT of q's autocorrelation
        # use scipy.welch to calculate loss 3 ######################
        # signal = q_pred_unbatched.T.cpu().detach().numpy()
        # _, q_pred_unbatched_fft = welch(signal, fs=200, nperseg=1024, nfft=fft_n)
        # q_pred_unbatched_fft = torch.from_numpy(q_pred_unbatched_fft).to(q_pred_unbatched.device)
        # q_pred_unbatched_fft = q_pred_unbatched_fft.requires_grad_(q_pred_unbatched.requires_grad)
           
        # sort out phi from low-order modes to high-order modes, based on the complexity of mode shapes
        # node = graph_unbatched[i].ndata['node']
        # phi_pred_bottom = phi_pred_unbatched[node[:, 1] == 0, :]
        # phi_pred_bottom = torch.concatenate((phi_pred_bottom[1:, :], phi_pred_bottom[:1, :]), 0)
        # curvature = torch.sum(torch.diff(phi_pred_bottom, n=1, dim=0)**2, dim=0)
        # _, sorted_indices = torch.sort(curvature, descending=False)
        # phi_pred_unbatched = phi_pred_unbatched[:, sorted_indices]
        
        # sort out q from low-order modes to high-order modes, based on the dominant frequency
        # _, q_pred_fft_max_indices = torch.max(q_pred_unbatched_fft.T, dim=0)
        # q_pred_sorted_indices = torch.argsort(q_pred_fft_max_indices)
        # q_pred_unbatched = q_pred_unbatched[:, q_pred_sorted_indices]
        
        if i == 0:
            acc_pred = phi_pred_unbatched @ q_pred_unbatched.T
            q_corr = torch.corrcoef(q_pred_unbatched.T)
            q_fft_corr = torch.corrcoef(q_pred_unbatched_fft)
            # q_mm = torch.mm(q_pred_unbatched.T, q_pred_unbatched)
            # q_fft_mm = torch.mm(q_pred_unbatched_fft, q_pred_unbatched_fft.T)
        else:
            acc_pred = torch.cat((acc_pred, phi_pred_unbatched @ q_pred_unbatched.T), 0)
            q_corr = torch.block_diag(q_corr, torch.corrcoef(q_pred_unbatched.T))
            q_fft_corr = torch.block_diag(q_fft_corr, torch.corrcoef(q_pred_unbatched_fft))
            # q_mm = torch.block_diag(q_mm, torch.mm(q_pred_unbatched.T, q_pred_unbatched))
            # q_fft_mm = torch.block_diag(q_fft_mm, torch.mm(q_pred_unbatched_fft, q_pred_unbatched_fft.T))
            
    batched_eye = torch.zeros_like(q_corr).fill_diagonal_(1)
    acc_true = graph[0].ndata['acc_Y']
    # incomplete measurements
    node_mask = graph[0].ndata['mask'] 
    acc_true = acc_true[node_mask, :]
    acc_pred = acc_pred[node_mask, :]
    # calculate loss
    
    loss1 = loss_func(acc_pred, acc_true)
    # loss1 = loss_func(acc_pred/acc_true, torch.ones_like(acc_true))  # normalized loss, does not work because some acc_true=0
    # loss1 = loss_func(torch.fft.rfft(acc_pred, n=fft_n).abs(), torch.fft.rfft(acc_true, n=fft_n).abs())
    
    loss2 = loss_func(q_corr, batched_eye)
    loss3 = loss_func(q_fft_corr, batched_eye)
    
    # loss2 = torch.norm(q_mm-batched_eye)
    # loss3 = torch.norm(q_fft_mm-batched_eye)
    
    loss4 = loss_func(phi_pred, graph[0].ndata['phi_Y'])
    
    # acc_pred_phase = torch.fft.rfft(acc_pred, n=fft_n).angle()
    # acc_true_phase = torch.fft.rfft(acc_true, n=fft_n).angle()
    # loss4 = loss_func(acc_pred_phase, acc_true_phase)
    return loss1, loss2, loss3, loss4