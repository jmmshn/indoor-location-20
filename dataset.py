# coding: utf-8
"""
Custom dataset for each building
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torch.autograd import Variable

from building import *

# custom dataset
class BuildingDataset(Dataset):
    
    def __init__(self, wifi, sensor, position, building_id):
        """
        Args:
            wifi     (array): [number of measurements, list of wifi ids, {freq type, dt}]
            sensor   (array): [number of measurements, {t, magn(3), acce(3), gyro(3)}]
            position (array): [number of measurements, {t, x, y, floor}]
            building_id (string): building ID
        """
        self.building_id = building_id
        
        # wifi type (0, 1, 2, 3)
        # use one hot encoding and then flatten across number of wifi's
        wifi_type = torch.tensor(wifi[:,:,0], dtype=torch.int64)
        self.wifi_type = F.one_hot(wifi_type, num_classes=4)
        self.wifi_type = self.wifi_type.view(self.wifi_type.shape[0], -1).to(dtype=torch.float)
        self.wifi_type = Variable(self.wifi_type, requires_grad=True)
        
        # wifi time intervals
        self.wifi_dt = torch.tensor(wifi[:,:,1], requires_grad=True, dtype=torch.float)
        
        # time of measurement
        self.time = torch.tensor(sensor[:,0])
        
        # sensor information (magn(3), acce(3), gyro(3))
        # standardize the values
        mean = np.mean(sensor[:,1:], axis=0)
        std = np.std(sensor[:,1:], axis=0)
        self.sensor = (sensor[:,1:] - mean) / std
        self.sensor = torch.tensor(self.sensor, requires_grad=True, dtype=torch.float)
        
        # truth x-y position 
        self.position = torch.tensor(position[:,1:3], requires_grad=True, dtype=torch.float)
        
        # truth floor position
        # use one hot encoding; floors cannot be negative
        floor_keys = BUILDING_INFO[self.building_id]['floor_dirs'].keys()
        floor_keys = [int(f) for f in floor_keys]
        self.min_floor = min(floor_keys)
        self.max_floor = max(floor_keys)
        num_floors = self.max_floor - self.min_floor + 1
        floor = torch.tensor(position[:,3], dtype=torch.int64) + abs(self.min_floor)
        self.floor = F.one_hot(floor, num_classes=num_floors).to(dtype=torch.float)
        self.floor = Variable(self.floor, requires_grad=True)
        
    def __len__(self):
        return self.time.shape[0]
    
    def __getitem__(self, idx):
        data = torch.cat((self.wifi_dt[idx,:], self.wifi_type[idx,:], self.sensor[idx,:]), axis=0)
        target = torch.cat((self.position[idx,:], self.floor[idx,:]), axis=0)
        t = self.time[idx]
        
        sample = {'data': data, 'target': target, 'time': t, 'idx': idx}
        return sample
        

