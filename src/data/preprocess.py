import os
import pandas as pd
import numpy as np

base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, '..', '..', 'data', 'unsw-nb15')
raw_dir = os.path.join(data_dir, 'raw')
processed_dir = os.path.join(data_dir, 'processed')

os.makedirs(processed_dir, exist_ok=True)

# import the dataset into pandas DataFrames
df_training = pd.read_csv(os.path.join(raw_dir, 'UNSW_NB15_training-set.csv'))
df_testing = pd.read_csv(os.path.join(raw_dir, 'UNSW_NB15_testing-set.csv'))
 
#Tag rows to ensure safe split later
df_training['set_marker'] = 'train'
df_testing['set_marker'] = 'test'

# stack the training and testing sets
df_data = pd.concat([df_training, df_testing], axis=0)

# remove the columns 'id' and 'attack_cat'
df_data.drop('id', inplace=True, axis=1)
df_data.drop('attack_cat', inplace=True, axis=1)

# 'is_ftp_login' should be a binary feature, we remove the instances that hold the values 2 and 4
df_data = df_data[df_data['is_ftp_login'] != 2]
df_data = df_data[df_data['is_ftp_login'] != 4]

df_data['ct_ftp_cmd'] = pd.to_numeric(df_data['ct_ftp_cmd'], errors='coerce').fillna(0)

categorical_features = ['state', 'service', 'proto']
df_data = pd.get_dummies(df_data, columns=categorical_features, prefix=categorical_features, prefix_sep=":")
# move the label back to the last column
df_data['label'] = df_data.pop('label')

# 7. Min-Max Normalization (Preventing Leakage)
continuous_features = ['dur', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 'sttl', 'dttl', 'sload', 'dload', 'sloss', 'dloss', 'sinpkt', 'dinpkt', 'sjit', 'djit', 'swin', 'stcpb', 'dtcpb', 'dwin', 'tcprtt', 'synack', 'ackdat', 'smean', 'dmean', 'trans_depth', 'response_body_len', 'ct_srv_src', 'ct_state_ttl', 'ct_dst_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'ct_ftp_cmd', 'ct_flw_http_mthd', 'ct_src_ltm', 'ct_srv_dst']

# Create a mask for training data to calculate statistics
train_mask = (df_data['set_marker'] == 'train')

# Calculate Min/Max ONLY on training data
min_vals = df_data.loc[train_mask, continuous_features].min()
max_vals = df_data.loc[train_mask, continuous_features].max()

# Apply to ALL data
df_data[continuous_features] = (df_data[continuous_features] - min_vals) / (max_vals - min_vals)

# Split back safely
df_training = df_data[df_data['set_marker'] == 'train'].drop('set_marker', axis=1)
df_testing = df_data[df_data['set_marker'] == 'test'].drop('set_marker', axis=1)

# Save
df_training.to_csv(os.path.join(processed_dir, 'train.csv'), index=False)
df_testing.to_csv(os.path.join(processed_dir, 'test.csv'), index=False)
