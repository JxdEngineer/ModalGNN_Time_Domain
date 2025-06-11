from models.models import create_model
from data.dataset_loader_numerical import get_dataset
from utils.loss_terms import loss_terms
from utils.test_metrics import test_metrics_single

import yaml

# pytorch lib
import os
import random
import torch
import dgl
import numpy as np
import time
from torch.optim.lr_scheduler import StepLR

# fix Seed - DGL cannot reproduce results by fixing the seed
seed = 123
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
dgl.seed(seed)
dgl.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ['CUDNN_DETERMINISTIC'] = '1'
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
torch.set_num_threads(1)


# Load config ########################################
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Desigante training device ########################################
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Training device:", device)


# Create datasets  ########################################
train_no = np.array(range(10))  # train the model on individual truss, one by one

start_time = time.time()
dataset_train = get_dataset(data_path=config['data']['path'], 
                        graph_no=train_no, 
                        time_0_list=config['shared']['time_0_list'], 
                        time_L=config['shared']['time_L'], 
                        mode_N=config['shared']['mode_N'])
dataloader_train = dgl.dataloading.GraphDataLoader(dataset_train, 
                              batch_size=1,
                              drop_last=False, shuffle=False)
print('Create training dataset: done')

dataset_time = (time.time() - start_time)
print("Create datasets: done, %s seconds" % dataset_time)
    

    
# Training loop ########################################   
c1 = 1
c2 = 1
c3 = 1

start_time = time.time()

freqs_id = []
phis_id_MAC = []
zetas_id = []

count = 0
for graph in dataloader_train:
    print('graph_id: ',count)
    # Create model  ########################################
    model = create_model(model_name=config['model']['name'],
                         dim=config['model']['dim'], 
                         time_L=config['shared']['time_L'], 
                         mode_N=config['shared']['mode_N'], 
                         dropout_rate=config['model']['dropout_rate'], 
                         hid_layer=config['model']['hid_layer'])

    print('Create model: done')
    
    # Setup optimizer ########################################
    optimizer = torch.optim.Adam(model.parameters(), lr=config['train']['learning_rate'])
    scheduler = StepLR(optimizer, step_size=config['train']['step_size'], 
                        gamma=config['train']['gamma'])
    # model training ##################################
    model.train()
    model.to(device)
    for epoch in range(config['train']['epochs']):
        graph_train = graph[0].to(device)
        # forward propogation ##################################
        q_pred_train, phi_pred_train = model(graph_train)
        loss1_train, loss2_train, loss3_train = \
            loss_terms(q_pred_train, phi_pred_train, graph_train,
                       fft_n=config['model']['fft_n'])
        loss_train = loss1_train*c1 + loss2_train*c2 + loss3_train*c3
        # back propogation ##################################
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
        scheduler.step()
        # monitor training process ##########################
        if (epoch + 1) % 100 == 0 or epoch == 0:
            print('epoch: {}, loss_train: {:.10f}' .format(epoch, loss_train))  
    count += 1 
    # model evaluation ##################################
    model.eval()
    freq_id, phi_id_MAC, zeta_id = test_metrics_single(graph_test=graph, mode_id_N=4, model_test=model)
    freqs_id.append(freq_id)
    zetas_id.append(zeta_id)
    phis_id_MAC.append(phi_id_MAC)
    
train_time = (time.time() - start_time)
print("Train the model: done, %s seconds" % train_time)

A_Freq_id = np.array(freqs_id)
A_Zeta_id = np.array(zetas_id)
A_Phi_MAC_id = np.array(phis_id_MAC)
