import os

coin_feat_dir = "full_npy"
file_list = os.listdir(coin_feat_dir)
for file in file_list:
    new_name = file.split("_", 1)[-1]
    new_name = new_name.split("_", 1)[-1]
    os.rename(os.path.join(coin_feat_dir, file), os.path.join(coin_feat_dir, new_name))