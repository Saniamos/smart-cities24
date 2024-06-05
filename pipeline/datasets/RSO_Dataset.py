import numpy as np
import pandas as pd
import torch
from torch.utils.data.dataset import Dataset
from io import BytesIO
from numpy.lib.npyio import NpzFile 

import os

marker_set = {'Ab',
              'Chest',
              'Head',
              'Hip',
              'LFArm',
              'LFoot',
              'LHand',
              'LShin',
              'LShoulder',
              'LThigh',
              'LToe',
              'LUArm',
              'Neck',
              'RFArm',
              'RFoot',
              'RHand',
              'RShin',
              'RShoulder',
              'RThigh',
              'RToe',
              'RUArm'}

class RSO_Dataset(Dataset):
    """Realsense Optitrack dataset."""
    
    def __init__(self, csv_file, root_dir,
                 transform=None, debug=False, in_mem=True, only_full_visible=False, fix_nan=True, **kwargs):
        
        self.root_dir = root_dir
        self.transform = transform
        self.nan_percentage = -1
        self.debug = debug
        self.nrows = 2000 if debug else None
        self.in_mem = in_mem
        self.fix_nan = fix_nan

        visible_path = os.path.join(self.root_dir, 'visibility.csv')
        self.only_full_visible = only_full_visible
        if self.only_full_visible:
            self.visible = pd.read_csv(visible_path, nrows=self.nrows)

        self.image_path_or_obj = os.path.join(self.root_dir, 'frames.npz')
        if self.in_mem:
            # load np file into cache instead of caching the getitem function for a more reliable and imediate cache size knowledge
            with open(self.image_path_or_obj, "rb") as fh:
                in_mem_arr = BytesIO(fh.read())
            self.images = NpzFile(in_mem_arr, own_fid=True, allow_pickle=True)
        else:
            self.images = np.load(self.image_path_or_obj)

        self.optitrack_data = self._load_and_init_optitrack_data(csv_file)

    def _load_and_init_optitrack_data(self, csv_file):
        # Load optitrack data from csv and prepare it
        optitrack_data, self.nan_percentage, self.bone_names = self._load_optitrack(csv_file, self.nrows, self.fix_nan)
        
        # existing frames in self.images:
        existing_frames = [int(x[6:]) for x in self.images.files]
        # remove frames that are not in the realsense recording
        optitrack_data = optitrack_data[optitrack_data.index.isin(existing_frames)]

        # Adjust number of samples so there are the same amount of realsense and optitrack entries.
        optitrack_data = optitrack_data[:len(self.images.files)]

        # remove frames due to visibility (this index already excludes skipped frames, so it needs to be performed afte the skip removal)
        if self.only_full_visible:
            optitrack_data = optitrack_data[self.visible['visibility'].to_numpy()[:len(optitrack_data)] > 0]

        return optitrack_data

    def __len__(self):
        return len(self.optitrack_data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # get entry by position (ie ith item which might have a higher index du to dropped frames)
        optitrack_entry = self.optitrack_data.iloc[idx]
        optitrack_point = optitrack_entry.to_numpy()
        
        # get entry by index
        realsense_image = self._load_image(optitrack_entry.name)

        # add channel dimension
        realsense_image = realsense_image[None, :]
        
        if self.transform:
            return self.transform(realsense_image, optitrack_point)
        return realsense_image, optitrack_point
    

    def _load_image(self, frame_number):
        # technically the frame is just float16, but that was only trouble, so we'll not be doing that anymore
        return self.images[f"frame_{frame_number:05}"].astype(np.float32) + 32768

    @staticmethod
    def _load_optitrack(csv_file, nrows, fix_nan):
        optitrack_data = pd.read_csv(csv_file, skiprows=2, header=[0, 1, 3, 4], index_col=0, nrows=nrows)
        optitrack_data = optitrack_data.sort_index(axis=1)
        optitrack_data = optitrack_data.rename(
            columns={'Unnamed: 1_level_0': "Time", 'Unnamed: 1_level_1': "Time", 'Unnamed: 1_level_2': "Time"})
        optitrack_data = optitrack_data.drop('Marker', axis=1)
        optitrack_data = optitrack_data.droplevel(level=0, axis=1)
        optitrack_data = optitrack_data.drop('Time', axis=1)
        column_list = optitrack_data.columns.to_list()
        column_list = [x for x in column_list if 'Rotation' in x]
        optitrack_data = optitrack_data.drop(columns=column_list)

        # rename Bones
        bone_names = [x.replace("Jonah Full Body:", "") for x in optitrack_data.columns.levels[0].to_list()]
        optitrack_data.columns = optitrack_data.columns.set_levels(bone_names, level=0)

        # calc percentage of NaN values
        isna = optitrack_data.isna().sum().sum()
        number_of_values = optitrack_data.shape[0] * optitrack_data.shape[1]
        nan_percentage = isna / number_of_values * 100
        
        # drop and fill empty values
        optitrack_data = optitrack_data.dropna(axis=1, how='all')
        if fix_nan:
            optitrack_data = optitrack_data.interpolate(axis=0, limit_direction='both')
        else:
            optitrack_data = optitrack_data.dropna(axis=0, how='any')

        return optitrack_data.astype(np.float32), nan_percentage, bone_names
    