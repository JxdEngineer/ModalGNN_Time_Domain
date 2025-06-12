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

epoch = df_loss_train['epoch'].values
loss_train_meter = df_loss_train['total_loss'].values
loss1_train_meter = df_loss_train['loss1'].values
loss2_train_meter = df_loss_train['loss2'].values
loss3_train_meter = df_loss_train['loss3'].values
time_train = df_loss_train['time'].values

loss_valid_meter = df_loss_valid['total_loss'].values
loss1_valid_meter = df_loss_valid['loss1'].values
loss2_valid_meter = df_loss_valid['loss2'].values
loss3_valid_meter = df_loss_valid['loss3'].values
time_valid = df_loss_valid['time'].values
# %% plot results
plt.close('all')
# plot log loss curve
plt.figure(constrained_layout=True)
plt.plot(epoch, loss_train_meter, label='train loss: '+f"{loss_train_meter[-1]:.6f}")
plt.plot(epoch, loss_valid_meter, label='valid loss: '+f"{loss_valid_meter[-1]:.6f}")
plt.ylabel('log(loss)', fontsize=14)
plt.xlabel('Epoch', fontsize=14)
title_text = "Total loss, Training Time={:.3f}".format(time_train[-1])
# plt.ylim([0.001, 10])
plt.title(title_text)
plt.grid()
plt.legend()
plt.show()