import torch
import torch.nn as nn
import dgl


# loss function using fft of modal responses
def loss_terms(q_pred, phi_pred, graph, fft_n):
    loss_func = nn.MSELoss()
    graph_unbatched = dgl.unbatch(graph[0])
    phi_index1 = 0
    phi_index2 = 0
    
    for i in range(len(graph_unbatched)):
        phi_index1 = phi_index2
        phi_index2 = phi_index1 + dgl.DGLGraph.number_of_nodes(graph_unbatched[i])
        q_pred_unbatched = q_pred[i, :, :]
        q_pred_unbatched_fft = torch.fft.rfft(q_pred_unbatched.T, n=fft_n).abs()
        phi_pred_unbatched = phi_pred[phi_index1:phi_index2]
        if i == 0:
            acc_pred = phi_pred_unbatched @ q_pred_unbatched.T
            q_corr = torch.corrcoef(q_pred_unbatched.T)
            q_fft_corr = torch.corrcoef(q_pred_unbatched_fft)
        else:
            acc_pred = torch.cat((acc_pred, phi_pred_unbatched @ q_pred_unbatched.T), 0)
            q_corr = torch.block_diag(q_corr, torch.corrcoef(q_pred_unbatched.T))
            q_fft_corr = torch.block_diag(q_fft_corr, torch.corrcoef(q_pred_unbatched_fft))  
    batched_eye = torch.zeros_like(q_corr).fill_diagonal_(1)
    acc_true = graph[0].ndata['acc_Y']
    # incomplete measurements
    node_mask = graph[0].ndata['mask'] 
    acc_true = acc_true[node_mask, :]
    acc_pred = acc_pred[node_mask, :]
    # calculate loss
    loss1 = loss_func(acc_pred, acc_true)
    loss2 = loss_func(q_corr, batched_eye)
    loss3 = loss_func(q_fft_corr, batched_eye)
    # loss4 = loss_func(phi_pred, graph[0].ndata['phi_Y'])
    
    acc_pred_phase = torch.fft.rfft(acc_pred, n=fft_n).angle()
    acc_true_phase = torch.fft.rfft(acc_true, n=fft_n).angle()
    loss4 = loss_func(acc_pred_phase, acc_true_phase)
    return loss1, loss2, loss3, loss4