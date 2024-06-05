from pqdm.threads import pqdm
from datasets.RSO_Dataset_Seq import RSO_Dataset_Seq
from datasets.RSO_LModule import RSO_LModule
from torch.utils.data import ConcatDataset

    
class RSO_LModule_Seq(RSO_LModule):
    def __init__(self, data_dir, batch_size, context=120, n_jobs=4, debug=False, in_mem=True, shuffle=True):
        super().__init__(data_dir=data_dir, batch_size=batch_size, n_jobs=n_jobs, debug=debug, in_mem=in_mem, shuffle=shuffle)
        self.context = context

    @property
    def data_shape(self):
        return (self.context, 1, 480, 848)

    def _load_datasets(self, sessions=[1, 2, 3, 4, 5], only_full_visible=False):
        # get all datasets with all available data (even half to no visible in realsense data)
        args = [dict(csv_file=f'{self.data_dir}session_{i}/optitrack/session_{i}.csv',
            root_dir=f'{self.data_dir}session_{i}/realsense/',
            in_mem=self.in_mem,
            debug=self.debug,
            context=self.context,
            only_full_visible=only_full_visible) for i in sessions]
        
        datasets = pqdm(args, RSO_Dataset_Seq, n_jobs=self.n_jobs, argument_type='kwargs') 
        
        # convenience in eval.ipynb / run_nopm.py
        self.columns = datasets[0].optitrack_data.columns
        
        return datasets
    
    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.train_data = ConcatDataset(self._load_datasets(sessions=[1, 2, 3, 4]))
            self.val_data = ConcatDataset(self._load_datasets(sessions=[5]))

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage == "predict":
            self.test_data = self._load_datasets(sessions=[6], only_full_visible=False)[0]
            self.test_data_full_vis = self._load_datasets(sessions=[6], only_full_visible=True)[0]
