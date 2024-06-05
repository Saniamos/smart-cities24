import lightning as pl
from pqdm.threads import pqdm
from datasets.RSO_Dataset import RSO_Dataset
from torch.utils.data import DataLoader, ConcatDataset

class RSO_LModule_noval(pl.LightningDataModule):
    # heavily based on: https://lightning.ai/docs/pytorch/stable/data/datamodule.html
    def __init__(self, data_dir, batch_size, n_jobs=4, debug=False, shuffle=True, in_mem=True, fix_nan=True):
        super().__init__()
        self.data_dir = data_dir
        self.debug = debug
        self.shuffle = shuffle
        self.n_jobs = n_jobs
        self.in_mem = in_mem
        self.batch_size = batch_size
        self.fix_nan = fix_nan
        self.data_loader_args = dict(batch_size=batch_size, 
                                     pin_memory=True, 
                                     num_workers=n_jobs)

    @property
    def data_shape(self):
        return (1, 480, 848)
    
    def num_data_loader(self, stage):
        if not self.debug and (stage == "test" or stage == "predict"):
            return 2
        return 1

    def _load_datasets(self, sessions=[1, 2, 3, 4, 5], only_full_visible=False):
        # get all datasets with all available data (even half to no visible in realsense data)
        args = [dict(csv_file=f'{self.data_dir}session_{i}/optitrack/session_{i}.csv',
            root_dir=f'{self.data_dir}session_{i}/realsense/',
            in_mem=self.in_mem,
            debug=self.debug,
            fix_nan=self.fix_nan,
            only_full_visible=only_full_visible) for i in sessions]
        
        datasets = pqdm(args, RSO_Dataset, n_jobs=self.n_jobs, argument_type='kwargs')
        
        # convenience in eval.ipynb / run_nopm.py
        self.columns = datasets[0].optitrack_data.columns
        
        return datasets

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.train_data = ConcatDataset(self._load_datasets(sessions=[1, 2, 3, 4, 5]))

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage == "predict":
            self.test_data = self._load_datasets(sessions=[6], only_full_visible=False)[0]
            self.test_data_full_vis = self._load_datasets(sessions=[6], only_full_visible=True)[0]

    def train_dataloader(self):
        return DataLoader(self.train_data, shuffle=self.shuffle, **self.data_loader_args)

    def test_dataloader(self):
        data_loader = DataLoader(self.test_data, **self.data_loader_args)
        if self.debug:
            return data_loader
        return data_loader, DataLoader(self.test_data_full_vis, **self.data_loader_args)

    def predict_dataloader(self):
        return self.test_dataloader()

    def teardown(self, stage: str) -> None:
        super().teardown(stage)
        if stage == "fit":
            self.train_data = None

        if stage == "test" or stage == "predict":
            self.test_data = None
            self.test_data_full_vis = None
