import torch
from torch import nn
import models.common as cm

"""
Genauso wie CNNModel_2022_06_22 nur ohne AveragePooling Layer

CNN aus Paper: 3D Human Pose Estimation from Monocular Images with Deep Convolutional Neural Network.
"""

name = "CNNModel_LiChan2014"

class NeuralNetwork(cm.NeuralNetwork):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.layers = nn.Sequential( # 848 x 480
            nn.Conv2d(1, 32, self.model_params["kernel_size"], self.model_params["stride"], self.model_params["padding"]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 424 x 240
            nn.Conv2d(32, 32, self.model_params["kernel_size"], self.model_params["stride"], self.model_params["padding"]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 212 x 120
            nn.Conv2d(32, 64, self.model_params["kernel_size"], self.model_params["stride"], self.model_params["padding"]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 106 x 60
            nn.LocalResponseNorm(32, self.model_params["local_response_norm_alpha"], self.model_params["local_response_norm_beta"]),
            nn.Conv2d(64, 64, self.model_params["kernel_size"], self.model_params["stride"], self.model_params["padding"]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 53 x 30
            nn.Flatten(),
            nn.Linear(53*30*64, 1024),
            nn.ReLU(),
            nn.Dropout(p=self.model_params["dropout_probability"]),
            nn.Linear(1024, 2048),
            nn.Tanh(),
            nn.Linear(2048, self.model_params["output_size"]),
        )


"""
Genauso wie CNNModel_LiChan2014

CNN aus Paper: 3D Human Pose Estimation from Monocular Images with Deep Convolutional Neural Network
Average Pooling als ersten Layern hinzugefügt
Nutzt Datensatz mit Knochen und Rotation
"""
hp_default = dict(
    loss_function = nn.MSELoss(),
    optimizer = torch.optim.Adam,

    data_params = cm.data_params,
    model_params = dict(
        dropout_probability = 0.25,
        kernel_size = 3,
        stride = 1,
        padding = 1,
        number_joints = 21,
        output_size = 63,
        local_response_norm_alpha = 0.0025,
        local_response_norm_beta = 0.75,
    ),
    
    scheduler_params = cm.scheduler_params,
    optimizer_params = cm.optimizer_params_adam,
    trainer_params = cm.trainer_params,
    early_stopping_params = cm.early_stopping_params,
    
)


"""
Same as above but with compositional loss
"""
hp_comp_loss = cm.adjust_for_compositional_loss(hp_default)

hp_l1_loss = cm.deepcopy(hp_default)
hp_l1_loss['loss_function'] = nn.L1Loss()

hp_dropna = cm.deepcopy(hp_l1_loss)
hp_dropna['data_params']['fix_nan'] = False