import torch
from torch import nn
import torchvision
import models.common as cm

"""
ResNet50 wie beschrieben im Paper: Compositional Human Pose Regression
"""

name = "ResNet50_SunShangetAl"

class NeuralNetwork(cm.NeuralNetwork):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        neural_network = torchvision.models.resnet50(weights=None)
        neural_network.fc = nn.Linear(2048, self.model_params['output_size'])
        neural_network.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.layers = neural_network


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
Same as above but with l1 loss
"""
hp_l1_loss = cm.deepcopy(hp_default)
hp_l1_loss['loss_function'] = nn.L1Loss()

# """
# Same as above but with compositional loss
# """
# hp_comp_loss = cm.adjust_for_compositional_loss(hp_default)

hp_dropna = cm.deepcopy(hp_l1_loss)
hp_dropna['data_params']['fix_nan'] = False


hp_orig = cm.deepcopy(hp_default)
hp_orig['loss_function'] = nn.L1Loss()
# hp_orig['eval_params']['distance'] = 30
del hp_orig['early_stopping_params']