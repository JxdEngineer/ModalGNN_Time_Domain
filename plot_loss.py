import pandas as pd
import matplotlib.pyplot as plt
import yaml

with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Read the log file into a pandas DataFrame
df_loss_train = pd.read_csv(config['model']['name']+"_loss_train.txt")
df_loss_valid = pd.read_csv(config['model']['name']+"_loss_valid.txt")

# Remove leading and trailing spaces from column names
df_loss_train.columns = df_loss_train.columns.str.strip()
df_loss_valid.columns = df_loss_valid.columns.str.strip()

loss_train_meter = df_loss_train['total_loss'].values
loss1_train_meter = df_loss_train['loss1'].values
loss2_train_meter = df_loss_train['loss2'].values
loss3_train_meter = df_loss_train['loss3'].values
loss4_train_meter = df_loss_train['loss4'].values
time_train = df_loss_train['time'].values

loss_valid_meter = df_loss_valid['total_loss'].values
loss1_valid_meter = df_loss_valid['loss1'].values
loss2_valid_meter = df_loss_valid['loss2'].values
loss3_valid_meter = df_loss_valid['loss3'].values
loss4_valid_meter = df_loss_valid['loss4'].values
time_valid = df_loss_valid['time'].values
# %% plot results
plt.close('all')
# plot log loss curve
plt.figure()
plt.semilogy(loss_train_meter, label='train loss: '+f"{loss_train_meter[-1]:.6f}")
plt.semilogy(loss_valid_meter, label='valid loss: '+f"{loss_valid_meter[-1]:.6f}")
plt.ylabel('MSE loss', fontsize=14)
plt.xlabel('Epoch', fontsize=14)
title_text = "Time={:.3f}".format(time_train[-1])
plt.ylim([0.001, 10])
plt.title(title_text)
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(1, 2, layout="constrained")
# plot different terms in the training loss
ax[0].semilogy(loss1_train_meter, label='Term 1: '+f"{loss1_train_meter[-1]:.6f}")
ax[0].semilogy(loss2_train_meter, label='Term 2: '+f"{loss2_train_meter[-1]:.6f}")
ax[0].semilogy(loss3_train_meter, label='Term 3: '+f"{loss3_train_meter[-1]:.6f}")
ax[0].semilogy(loss4_train_meter, label='Term 4: '+f"{loss4_train_meter[-1]:.5f}")
# ax[0].semilogy(loss5_train_meter, label='Term 5: '+f"{loss5_train_meter[-1]:.5f}")
title_text = "Training loss"
ax[0].set_title(title_text)
ax[0].set_ylabel('log(Loss)', fontsize=14)
ax[0].set_xlabel('Epoch', fontsize=14)
ax[0].set_ylim([0.000001, 10])
ax[0].grid()
ax[0].legend()
# plot different terms in the validation loss
ax[1].semilogy(loss1_valid_meter, label='Term 1: '+f"{loss1_valid_meter[-1]:.6f}")
ax[1].semilogy(loss2_valid_meter, label='Term 2: '+f"{loss2_valid_meter[-1]:.6f}")
ax[1].semilogy(loss3_valid_meter, label='Term 3: '+f"{loss3_valid_meter[-1]:.6f}")
ax[1].semilogy(loss4_valid_meter, label='Term 4: '+f"{loss4_valid_meter[-1]:.5f}")
# ax[1].semilogy(loss5_valid_meter, label='Term 5: '+f"{loss5_valid_meter[-1]:.5f}")
title_text = "Validation loss"
ax[1].set_title(title_text)
ax[1].set_ylabel('log(Loss)', fontsize=14)
ax[1].set_xlabel('Epoch', fontsize=14)
ax[1].set_ylim([0.000001, 10])
ax[1].grid()
ax[1].legend()