import torch
from torch import nn
import models.common as cm

"""
Selbst designtes CNN
Nutzt Datensatz mit Knochen und Rotation
Daten nicht zufällig.
"""

name = "OwnBaselineCNN"

class NeuralNetwork(cm.NeuralNetwork):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=3),  # 282*160
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=3),  # 94*53
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 47*26
            nn.Flatten(),
            nn.Linear(47*26*32, 4096),
            nn.ReLU(),
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, self.model_params["output_size"]),
        )

"""
Default CNN HP using MSE loss and Adam
"""
hp_default = dict(
    loss_function = nn.MSELoss(),
    optimizer = torch.optim.Adam,

    data_params = {**cm.data_params, "batch_size": 46},
    model_params = cm.model_params,
    
    scheduler_params = cm.scheduler_params,
    optimizer_params = cm.optimizer_params_adam,
    trainer_params = cm.trainer_params,
    early_stopping_params = cm.early_stopping_params,
    
)

"""
Same as above but with compositional loss
"""
hp_comp_loss = cm.adjust_for_compositional_loss(hp_default, use_sgd=False)

hp_l1_loss = cm.deepcopy(hp_default)
hp_l1_loss['loss_function'] = nn.L1Loss()

hp_dropna = cm.deepcopy(hp_l1_loss)
hp_dropna['data_params']['fix_nan'] = False


# just to have different naming when running a different data source
hp_orig = cm.deepcopy(hp_default)
hp_orig['trainer_params']['max_epochs'] = 50
# hp_orig['eval_params']['distance'] = 20
del hp_orig['early_stopping_params']