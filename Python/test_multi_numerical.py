# pytorch lib
import torch
import numpy as np
import time
import dgl
import matplotlib.pyplot as plt

from scipy.signal import welch

from models.models import create_model
from data.dataset_loader_numerical import get_dataset
from utils.match_mode import MAC
from utils.match_mode import match_mode
from utils.test_metrics import test_metrics_single
import yaml

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

model.eval()

# designate sample no. for testing ######################################
test_no = np.array(range(100)) 
dataset_test = get_dataset(data_path=config['data']['path'], 
                        graph_no=test_no, 
                        time_0_list=config['shared']['time_0'], 
                        time_L=config['shared']['time_L'], 
                        mode_N=config['shared']['mode_N'])
dataloader_test = dgl.dataloading.GraphDataLoader(dataset_test, 
                              batch_size=1,
                              drop_last=False, shuffle=False)
print('Create dataset: done')
# %% calculate test results
mode_id_N = 4

N_test = len(test_no)
plt.close('all')
freqs_id = []
phis_id_MAC = []
zetas_id = []

count = 0
start_time = time.time()

for graph in dataloader_test:
    freq_id, phi_id_MAC, zeta_id = test_metrics_single(graph_test=graph, mode_id_N=4, model_test=model)
    freqs_id.append(freq_id)
    zetas_id.append(zeta_id)
    phis_id_MAC.append(phi_id_MAC)
    # print(graph)

test_time = (time.time() - start_time)
print("Test the model: done, %s seconds" % test_time)   

AAA_Freq_id = np.array(freqs_id)
AAA_Zeta_id = np.array(zetas_id)
AAA_Phi_MAC_id = np.array(phis_id_MAC)