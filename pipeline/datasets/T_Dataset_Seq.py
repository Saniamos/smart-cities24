import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import pandas as pd

class T_Dataset_Seq(Dataset):
    """Realsense Optitrack dataset."""
    
    def __init__(self, data_dir, file, session, transform=None, debug=False, context=120):
        self.transform = transform
        self.context = context
        self.data_dir = data_dir
        self.file = file
        
        nrows = 2000 if debug else None
    
        prd = pd.read_parquet(f"{data_dir}transformed/{file}.prd.parquet")
        self.prd = prd[prd['Session'] == session][:nrows].drop('Session', axis=1).to_numpy().astype(np.float32)
        
        trg = pd.read_parquet(f"{data_dir}transformed/{file}.trg.parquet")
        self.trg = trg[trg['Session'] == session][:nrows].drop('Session', axis=1).to_numpy().astype(np.float32)
    
        self.columns = prd.drop('Session', axis=1).columns
    
    def __len__(self):
        # shift the length, so that 0 is the first index that can be accessed but correspontds to image 120 (if context is 120)
        return self.prd.shape[0] - self.context

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data = self.prd[idx:idx+self.context]
        target = self.trg[idx+self.context]

        if self.transform:
            return self.transform(data, target)
        return data, target
