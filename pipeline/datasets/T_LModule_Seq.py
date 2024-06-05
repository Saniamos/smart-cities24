from pqdm.threads import pqdm
from datasets.T_Dataset_Seq import T_Dataset_Seq
import lightning as pl
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader, ConcatDataset

    
class T_LModule_Seq(pl.LightningDataModule):
    def __init__(self, data_dir, file,  batch_size, context=120, n_jobs=4, debug=False, shuffle=True):
        super().__init__()
        self.context = context
        self.debug = debug
        self.shuffle = shuffle

        self.data_dir = data_dir
        self.file = file

        self.n_jobs = n_jobs

        self.data_loader_args = dict(batch_size=batch_size, 
                                pin_memory=True, 
                                num_workers=n_jobs)

    @property
    def data_shape(self):
        return (self.context, 63)
    
    def num_data_loader(self, stage):
        return 1
    
    def _load_datasets(self, sessions=[1, 2, 3, 4, 5]):
        # get all datasets with all available data (even half to no visible in realsense data)
        args = [dict(file=self.file,
            data_dir=self.data_dir,
            session=i,
            context=self.context,
            debug=self.debug) for i in sessions]
        
        datasets = pqdm(args, T_Dataset_Seq, n_jobs=self.n_jobs, argument_type='kwargs')
        
        # convenience in eval.ipynb / run_nopm.py
        self.columns = datasets[0].columns
        
        return datasets

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.train_data = ConcatDataset(self._load_datasets(sessions=[1, 2, 3, 4]))
            self.val_data = ConcatDataset(self._load_datasets(sessions=[5]))

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage == "predict":
            self.test_data = self._load_datasets(sessions=[6])[0]

    def train_dataloader(self):
        return DataLoader(self.train_data, shuffle=self.shuffle, **self.data_loader_args)

    def val_dataloader(self):
        return DataLoader(self.val_data, **self.data_loader_args)

    def test_dataloader(self):
        return DataLoader(self.test_data, **self.data_loader_args)

    def predict_dataloader(self):
        return self.test_dataloader()

    def teardown(self, stage: str) -> None:
        super().teardown(stage)
        if stage == "fit":
            self.train_data = None
            self.val_data = None

        if stage == "test" or stage == "predict":
            self.test_data = None
