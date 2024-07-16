import torch
import torch.utils.data as data
import h5py
import numpy as np
import os
import random
from utils import LDRtoHDR, ColorAugmentation


# LDR bit
dr = float(2**16 - 1)

################################  Read Training data  ################################
class TrainSetLoader(data.Dataset):
    def __init__(self, cfg):
        super(TrainSetLoader, self).__init__()
        self.dataset_dir = cfg.trainset_path
        file_list = os.listdir(cfg.trainset_path)
        item_num = len(file_list)
        self.item_num = item_num

    def __getitem__(self, index):
        dataset_dir = self.dataset_dir
        index = index + 1
        file_name = [dataset_dir + '/%06d' % index + '.h5']
        with h5py.File(file_name[0], 'r') as hf:
            # ldr_data, hdr_label: [ah,aw,h,w,3];  ev_data: [3,1]
            ldr_data1 = np.array(hf.get('data1')).astype(np.float32) / dr
            ldr_data2 = np.array(hf.get('data2')).astype(np.float32) / dr
            ldr_data3 = np.array(hf.get('data3')).astype(np.float32) / dr
            hdr_label = np.array(hf.get('label'))
            ev_data = np.array(hf.get('ev_data'))
            ev_data = 2 ** ev_data

            # [ah,aw,h,w,9]
            ldr_data = np.concatenate((ldr_data1, ldr_data2, ldr_data3), axis=-1)
      
            # 4D Augmentation
            # flip
            if np.random.rand(1) > 0.5:
                ldr_data = np.flip(np.flip(ldr_data, 0), 2)
                hdr_label = np.flip(np.flip(hdr_label, 0), 2)
            if np.random.rand(1) > 0.5:
                ldr_data = np.flip(np.flip(ldr_data, 1), 3)
                hdr_label = np.flip(np.flip(hdr_label, 1), 3)
            # rotate
            r_ang = np.random.randint(1, 5)
            ldr_data = np.rot90(ldr_data, r_ang, (2, 3))
            ldr_data = np.rot90(ldr_data, r_ang, (0, 1))
            hdr_label = np.rot90(hdr_label, r_ang, (2, 3))
            hdr_label = np.rot90(hdr_label, r_ang, (0, 1))

            # Get input and label
            in_data = np.transpose(ldr_data, [4, 0, 1, 2, 3])         # [9,ah,aw,h,w]
            expo_data = ev_data                                       # [3,1]
            in_label = np.transpose(hdr_label, [4, 0, 1, 2, 3])       # [3,ah,aw,h,w]

            # Convert to tensor
            in_data = torch.from_numpy(in_data.copy())
            expo_data = torch.from_numpy(expo_data.copy())
            in_label = torch.from_numpy(in_label.copy())

        return in_data, expo_data, in_label

    def __len__(self):
        return self.item_num


################################  Read Test data  ################################
class TestSetLoader(data.Dataset):
    def __init__(self, cfg):
        super(TestSetLoader, self).__init__()
        self.dataset_dir = cfg.testset_path
        file_list = os.listdir(cfg.testset_path)
        item_num = len(file_list)
        self.item_num = item_num

    def __getitem__(self, index):
        dataset_dir = self.dataset_dir
        index = index + 1
        file_name = [dataset_dir + '/%03d' % index + '.h5']
        with h5py.File(file_name[0], 'r') as hf:
            # ldr_data, hdr_label: [ah,aw,h,w,3];  ev_data: [3,1]
            ldr_data1 = np.array(hf.get('data1')).astype(np.float32) / dr
            ldr_data2 = np.array(hf.get('data2')).astype(np.float32) / dr
            ldr_data3 = np.array(hf.get('data3')).astype(np.float32) / dr
            hdr_label = np.array(hf.get('label'))
            ev_data = np.array(hf.get('ev_data'))
            ev_data = 2 ** ev_data

            # [ah,aw,h,w,9]
            ldr_data = np.concatenate((ldr_data1, ldr_data2, ldr_data3), axis=-1)

            # Get input and label
            in_data = np.transpose(ldr_data, [4, 0, 1, 2, 3])       # [9,ah,aw,h,w]
            expo_data = ev_data                                     # [3,1]
            in_label = np.transpose(hdr_label, [4, 0, 1, 2, 3])     # [3,ah,aw,h,w]

            # Convert to tensor
            in_data = torch.from_numpy(in_data.copy())
            expo_data = torch.from_numpy(expo_data.copy())
            in_label = torch.from_numpy(in_label.copy())

        return in_data, expo_data, in_label

    def __len__(self):
        return self.item_num