from typing import Any
import torch
import numpy as np
class data_loader (torch.utils.data.Dataset):
    def __init__(self,data,label):
        self.data = data
        self.label = label
    
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        genotype = self.data[index].permute(2,0,1).float()
        label = self.label[index].float()
        return genotype,label
