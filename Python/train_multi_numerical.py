from models.models import create_model
from data.dataset_loader_numerical import get_dataset
from utils.loss_terms import loss_terms
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

def train(config_path):
    # Load config ########################################
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Desigante training device ########################################
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Training device:", device)
    
    # Create model  ########################################
    model = create_model(model_name=config['model']['name'],
                         dim=config['model']['dim'], 
                         time_L=config['shared']['time_L'], 
                         mode_N=config['shared']['mode_N'], 
                         dropout_rate=config['model']['dropout_rate'], 
                         hid_layer=config['model']['hid_layer'])
    model.to(device)
    print('Create model: done')
    
    # Create datasets  ########################################
    train_no = np.array(range(80))
    valid_no = np.array(range(80, 90))
    
    start_time = time.time()
    dataset_train = get_dataset(data_path=config['data']['path'], 
                            graph_no=train_no, 
                            time_0_list=config['shared']['time_0_list'], 
                            time_L=config['shared']['time_L'], 
                            mode_N=config['shared']['mode_N'])
    dataloader_train = dgl.dataloading.GraphDataLoader(dataset_train, 
                                  batch_size=config['data']['bs'],
                                  drop_last=False, shuffle=True)
    print('Create training dataset: done')
    dataset_valid = get_dataset(data_path=config['data']['path'], 
                            graph_no=valid_no, 
                            time_0_list=config['shared']['time_0'], 
                            time_L=config['shared']['time_L'], 
                            mode_N=config['shared']['mode_N'])
    dataloader_valid = dgl.dataloading.GraphDataLoader(dataset_valid, 
                                  batch_size=config['data']['bs'],
                                  drop_last=False, shuffle=False)
    print('Create validation dataset: done')
    dataset_time = (time.time() - start_time)
    print("Create datasets: done, %s seconds" % dataset_time)
    
    
    # Setup optimizer ########################################
    optimizer = torch.optim.Adam(model.parameters(), lr=config['train']['learning_rate'])
    scheduler = StepLR(optimizer, step_size=config['train']['step_size'], 
                        gamma=config['train']['gamma'])
        
    # Path to log file ########################################
    log_file_path_train = config['model']['name'] + "_loss_train.txt"
    with open(log_file_path_train, 'w') as f:
        f.write("epoch, time, loss1, loss2, loss3, total_loss\n")  # Header for the log file
    log_file_path_valid = config['model']['name'] + "_loss_valid.txt"
    with open(log_file_path_valid, 'w') as f:
        f.write("epoch, time, loss1, loss2, loss3, total_loss\n")  # Header for the log file
 
    # Training loop ########################################   
    c1 = 10
    c2 = 1
    c3 = 1
       
    start_time = time.time()
    for epoch in range(config['train']['epochs']):
        
        # model training ##################################
        model.train()
        epoch_loss_train = 0
        
        for graph_train in dataloader_train:
            graph_train = graph_train[0].to(device)
            q_pred_train, phi_pred_train = model(graph_train)  # model inference
            # phi_pred_train = graph_train[0].ndata['phi_Y']  # use true phi for training 
            loss1_train, loss2_train, loss3_train = \
                loss_terms(q_pred_train, phi_pred_train, graph_train,
                            fft_n=config['model']['fft_n'])
            loss_train = loss1_train*c1 + loss2_train*c2 + loss3_train*c3
            # backpropogation ##################################
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()
            epoch_loss_train += loss_train
        
        scheduler.step()
        # scheduler.step(epoch_loss_train)
        epoch_loss_train /= len(dataloader_train)
        train_time = (time.time() - start_time)
        # Write training losses to the log file
        with open(log_file_path_train, 'a') as f:
            f.write(f"{epoch}, {train_time:.3f}, {loss1_train:.10f}, {loss2_train:.10f}, {loss3_train:.10f}, {loss_train:.10f}\n")

        # epoch_loss_valid = epoch_loss_train # when no validation is used
        # model validation ##################################
        model.eval()
        epoch_loss_valid = 0
        with torch.no_grad(): # Turn off gradient tracking
            for graph_valid in dataloader_valid:
                graph_valid = graph_valid[0].to(device)
                q_pred_valid, phi_pred_valid = model(graph_valid)  # model inference
                # phi_pred_valid = graph_valid[0].ndata['phi_Y']  # use true phi for training 
                loss1_valid, loss2_valid, loss3_valid = \
                    loss_terms(q_pred_valid, phi_pred_valid, graph_valid, 
                                fft_n=config['model']['fft_n'])
                loss_valid = loss1_valid*c1 + loss2_valid*c2 + loss3_valid*c3
                epoch_loss_valid += loss_valid
        epoch_loss_valid /= len(dataloader_valid)
        train_time = (time.time() - start_time)
        # Write validation losses to the log file
        with open(log_file_path_valid, 'a') as f:
            f.write(f"{epoch}, {train_time:.3f}, {loss1_valid:.10f}, {loss2_valid:.10f}, {loss3_valid:.10f}, {loss_valid:.10f}\n")

        # print losses
        if (epoch + 1) % 100 == 0 or epoch == 0:
            print('epoch: {}, loss_train: {:.10f}, loss_valid: {:.10f}' 
                  .format(epoch, epoch_loss_train, epoch_loss_valid))  
    
    train_time = (time.time() - start_time)
    print("Train the model: done, %s seconds" % train_time)
    
    # Save the model's state dictionary ########################################
    model_save_path = config['model']['name'] + '.pth'
    torch.save(model.state_dict(), model_save_path)
    print('Save trained model: done')

if __name__ == "__main__":
    train('config/config.yaml')