import os
import torch
import numpy as np

from torch.utils.data import Dataset, TensorDataset


class Mydataset(Dataset):
    def __init__(self, img_path, label_path, label_length_path, is_train=True, transform=None):
        self.path = img_path
        self.label_path = label_path
        self.label_length_path = label_length_path
        self.transform = transform
        self.is_train = is_train
        
        self.img = os.listdir(self.path)
        self.labels = open(self.label_path, 'r').read().split('\n')
        self.labels_length = open(self.label_length_path, 'r').read().split('\n')

    def __getitem__(self, idx):
        img = np.load(f'{self.path}/{idx}.npy')  # (input_seq, H, W)
        label = torch.tensor(np.array(self.labels[idx].split(), dtype=int))
        label_length = torch.tensor([int(self.labels_length[idx])])
        img_out = torch.zeros(img.shape)
        if self.transform is not None:
            for i in range(img.shape[0]):
                img_seg = img[i, :, :]  # (H, W)
                img_transform = self.transform(img_seg.astype(np.uint8))  # (H, W) -> (1, H, W)
                img_out[i, :, :] = img_transform.squeeze(0)  # (1, H, W) -> (H, W)
        else:
            img_out = torch.tensor(img)
        return img_out, label, label_length

    def __len__(self):
        return len(self.img)


def Mytensordataset(img_path, label_path, label_length_path, is_train=True, transform=None):
    tensor_img = torch.from_numpy(torch.load(img_path))
    tensor_label = torch.from_numpy(torch.load(label_path))
    tensor_label_length = torch.from_numpy(torch.load(label_length_path))
    if transform is not None:
        H = tensor_img.size(-2)
        W = tensor_img.size(-1)
        T = tensor_img.size(1)
        tensor_img = tensor_img.view(-1, H, W).unsqueeze(1)  # (Nxinput_seq, 1(C), H, W)
        for i in range(tensor_img.size(0)):
            tensor_img[i, 0, :, :] = transform(tensor_img[i, 0, :, :].data.numpy().astype(np.uint8)).squeeze(0)
        tensor_img = tensor_img.squeeze(1).view(-1, T, H, W).type(torch.float)
    
    return TensorDataset(tensor_img, tensor_label, tensor_label_length)  # (N, input_seq, H, W), (NXoutput_seq), (N, )


def collate_fn(batch):
    img = torch.cat([item[0] for item in batch], dim=0)
    label = torch.cat([item[1] for item in batch], dim=0)
    label_length = torch.cat([item[2] for item in batch], dim=0)
    img = img.unsqueeze(1)  # (Bxinput_seq, C=1, H, W)
    return img, label, label_length