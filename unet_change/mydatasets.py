from torch.utils.data.dataset import Dataset
import numpy as np
import torch

def load_data(dt_path):

    dt=np.load(dt_path)#800*32*32*3
    dt=(dt-np.min(dt))/(np.max(dt)-np.min(dt))  # 最值归一化
    dt=torch.tensor(dt)
    dt=dt.permute(0,3,1,2)#800*3*32*32
    return dt

'''
class CreateDatasets(Dataset):
    def __init__(self, spin_path, hd_path):

        self.spin = load_data(spin_path)
        self.hd = load_data(hd_path)
        assert(self.spin.size(0)==self.hd.size(0))

    def __len__(self):
        return len(self.spin)

    def __getitem__(self, item):
        return self.spin[item], self.hd[item]
'''

class CreateDatasets(Dataset):
    def __init__(self, input_path,label_path):

        self.input = load_data(input_path)#800*3*32*32
        self.label = load_data(label_path)
        #assert(self.spin.size(0)==self.hd.size(0))

    def __len__(self):
        return len(self.input)

    def __getitem__(self, item):
        return self.input[item], self.label[item]#3*32*32
       