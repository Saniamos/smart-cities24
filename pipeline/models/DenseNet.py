import torch
from torch import nn
import torchvision
import models.common as cm



class NeuralNetwork(cm.NeuralNetwork):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # neural_network = torchvision.models.densenet161(weights=None)
        # neural_network.classifier = nn.Linear(2208, self.model_params['output_size'])
        # neural_network.features.conv0 = nn.Conv2d(1, 96, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        neural_network = torchvision.models.densenet121(weights=None)
        neural_network.classifier = nn.Linear(1024, self.model_params['output_size'])
        neural_network.features.conv0 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        self.layers = neural_network

hp_default = dict(
    loss_function = nn.MSELoss(),
    optimizer = torch.optim.Adam,

    data_params = {**cm.data_params, "batch_size": 8},
    model_params = cm.model_params,
    
    scheduler_params = cm.scheduler_params,
    optimizer_params = cm.optimizer_params_adam,
    trainer_params = cm.trainer_params,
    early_stopping_params = cm.early_stopping_params,
    
)


hp_l1_loss = cm.deepcopy(hp_default)
hp_l1_loss['loss_function'] = nn.L1Loss()


hp_dropna = cm.deepcopy(hp_l1_loss)
hp_dropna['data_params']['fix_nan'] = False