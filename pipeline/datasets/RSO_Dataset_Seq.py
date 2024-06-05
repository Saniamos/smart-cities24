import numpy as np
import torch
from datasets.RSO_Dataset import RSO_Dataset

class RSO_Dataset_Seq(RSO_Dataset):
    """Realsense Optitrack dataset."""
    
    def __init__(self, csv_file, root_dir,
                 transform=None, debug=False, in_mem=True, only_full_visible=True, context=120):
        super().__init__(csv_file=csv_file, root_dir=root_dir, transform=None, debug=debug, in_mem=in_mem, only_full_visible=only_full_visible)
        self.transform_seq = transform
        self.context = context
    
    def __len__(self):
        # shift the length, so that 0 is the first index that can be accessed but correspontds to image 120 (if context is 120)
        return super().__len__() - self.context

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        realsense_image = []
        optitrack_point = None
        # because the rso is not shuffled, the index is in the right order (checked in vis_data.ipynb)
        for i in range(idx, idx + self.context):
            rso = super().__getitem__(i)
            realsense_image.append(rso[0])
            optitrack_point = rso[1]
            
        realsense_image = np.array(realsense_image)

        if self.transform_seq:
            return self.transform_seq(realsense_image, optitrack_point)
        return realsense_image, optitrack_point
