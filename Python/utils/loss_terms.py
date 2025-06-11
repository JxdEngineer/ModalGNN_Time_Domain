import torch
import torch.nn as nn
import dgl

# loss function using fft of modal responses - sum of individual calculation
def loss_terms(q_pred, phi_pred, graph, fft_n):
    loss_func = nn.MSELoss()
    graph_unbatched = dgl.unbatch(graph)
    loss1, loss2, loss3 = 0, 0, 0
    
    phi_index1, phi_index2 = 0, 0
    
    for i in range(len(graph_unbatched)):
        phi_index1 = phi_index2
        phi_index2 = phi_index1 + dgl.DGLGraph.number_of_nodes(graph_unbatched[i])
        phi_pred_unbatched = phi_pred[phi_index1:phi_index2]
           
        q_pred_unbatched = q_pred[i, :, :]
        # use FFT of q to calculate loss 3 ######################
        q_pred_unbatched_fft = torch.fft.rfft(q_pred_unbatched.T, n=fft_n).abs()
        
        acc_pred = phi_pred_unbatched @ q_pred_unbatched.T
        acc_true = graph_unbatched[i].ndata['acc_Y']
        
        q_corr = torch.corrcoef(q_pred_unbatched.T)
        q_fft_corr = torch.corrcoef(q_pred_unbatched_fft)
        I = torch.zeros_like(q_corr).fill_diagonal_(1)
        
        loss1 += loss_func(acc_pred, acc_true)
        loss2 += loss_func(q_corr, I)
        loss3 += loss_func(q_fft_corr, I)
        
    loss1 /= len(graph_unbatched)
    loss2 /= len(graph_unbatched)
    loss3 /= len(graph_unbatched)
    
    return loss1, loss2, loss3