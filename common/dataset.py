import torch
import os
import numpy as np


class SpeechCommandsDataset(torch.utils.data.Dataset):
  def __init__(self, data_dir, split):
    assert split in ['train', 'valid', "test", "train_ssl"]
    assert os.path.exists(data_dir)
    super().__init__()
    labels_path = os.path.join(data_dir, split, "labels.pth")
    features_path = os.path.join(data_dir, split, "features.pth")
    self.features = torch.load(features_path, weights_only=True)
    self.labels = torch.load(labels_path, weights_only=True).long()
    print("FEATURES SHAPE:", self.features.shape)
    # shuffle the data
    indices = np.random.permutation(len(self.features))
    self.features = self.features[indices]
    self.labels = self.labels[indices]

  def __getitem__(self, index):
    f = self.features[index].unsqueeze(0)
    f = (f - f.mean()) / (f.std()+1e-6)
    lbl = self.labels[index]
    return f, lbl

  def __len__(self):
    return len(self.features)


def setup_data(base_data_dir, file_id=None, BATCH_SIZE=128):
  import subprocess as sp
  import gdown

  data_dir = os.path.join(base_data_dir, "speechcommands_ssl_data")
  tar_path = os.path.join(base_data_dir, "speechcommands_ssl_data.tar.gz")

  if not os.path.exists(data_dir):
    if not os.path.exists(tar_path):
      print("Downloading data..")
      gdown.download(id=file_id)
    else:
      print("Found tar file..")
    print("extracting data..")
    command = f"cd {base_data_dir} && tar xf speechcommands_ssl_data.tar.gz && cd -"
    print(command)
    sp.call(command, shell=True)
  else:
    print(f"{data_dir} exists..")

  train_dset = SpeechCommandsDataset(data_dir, "train")
  train_ssl_dset = SpeechCommandsDataset(data_dir, "train_ssl")
  val_dset = SpeechCommandsDataset(data_dir, "valid")
  test_dset = SpeechCommandsDataset(data_dir, "test")

  tr_dataloader = torch.utils.data.DataLoader(train_dset, batch_size=BATCH_SIZE, num_workers=4, shuffle=True)
  tr_ssl_dataloader = torch.utils.data.DataLoader(train_ssl_dset, batch_size=BATCH_SIZE, num_workers=4, shuffle=True)
  val_dataloader = torch.utils.data.DataLoader(val_dset, batch_size=BATCH_SIZE, num_workers=4, shuffle=False)
  test_dataloader = torch.utils.data.DataLoader(test_dset, batch_size=BATCH_SIZE, num_workers=4, shuffle=False)
  return tr_dataloader, val_dataloader, test_dataloader, tr_ssl_dataloader, train_ssl_dset, train_dset
