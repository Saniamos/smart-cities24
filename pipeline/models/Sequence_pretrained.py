import torch
from torch import nn
import models.common as cm
import importlib


class NeuralNetwork(cm.NeuralNetwork):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        VisionModule = importlib.import_module(f"models.{self.model_params['vision_model_lib']}")
        self.vision_model = VisionModule.NeuralNetwork.load_from_checkpoint(self.model_params['vision_model_path'])

        SequenceModule = importlib.import_module(f"models.{self.model_params['sequence_model_lib']}")
        self.sequence_model = SequenceModule.NeuralNetwork.load_from_checkpoint(self.model_params['sequence_model_path'])

        if not self.model_params['retrain']:
            # Freeze all vision layers
            for param in self.vision_model.parameters():
                param.requires_grad = False
                        
            # Freeze all sequence layers
            for param in self.sequence_model.parameters():
                param.requires_grad = False

            # Unfreeze the last layer
            for param in self.sequence_model.fc.parameters():
                param.requires_grad = True

    # input dimensions are: batch, seq, pixel, height, width
    def forward(self, x_3d):
        batch, seq, pix, height, width = x_3d.shape
        out = self.vision_model(x_3d.view(batch * seq, pix, height, width)).view(batch, seq, -1)
        return self.sequence_model(out)
    
    
hp_default = dict(
    loss_function = nn.MSELoss(),
    optimizer = torch.optim.Adam,

    data_params = dict(
        batch_size=16,
        context= 15, 
        shuffle= True
    ),

    model_params = dict(
        vision_model_lib = 'ResNet50_SunShangetAl',
        vision_model_path = './checkpoints/ResNet50_SunShangetAl.hp_l1_loss.ckpt',
        sequence_model_lib = 'T_Sequence',
        sequence_model_path = './checkpoints/T_Sequence.hp_l1_loss.ckpt',
        retrain = False,
    ),

    scheduler_params = cm.scheduler_params,
    optimizer_params = cm.optimizer_params_adam,
    trainer_params = dict(
        max_epochs = 1,
        precision = "32",
    ),
    early_stopping_params = cm.early_stopping_params,
    
)

hp_l1_loss = cm.deepcopy(hp_default)
hp_l1_loss['loss_function'] = nn.L1Loss()

hp_l1_loss_retrain = cm.deepcopy(hp_default)
hp_l1_loss_retrain['loss_function'] = nn.L1Loss()
hp_l1_loss_retrain['model_params']['retrain'] = True
hp_l1_loss_retrain['trainer_params']['max_epochs'] = 20

hp_OwnBaselineCNN = cm.deepcopy(hp_default)
hp_OwnBaselineCNN['model_params']['vision_model_lib'] = 'OwnBaselineCNN'
hp_OwnBaselineCNN['model_params']['vision_model_path'] = './checkpoints/OwnBaselineCNN.hp_default.ckpt'

hp_CNNModel_LiChan2014 = cm.deepcopy(hp_default)
hp_CNNModel_LiChan2014['model_params']['vision_model_lib'] = 'CNNModel_LiChan2014'
hp_CNNModel_LiChan2014['model_params']['vision_model_path'] = './checkpoints/CNNModel_LiChan2014.hp_default.ckpt'

hp_CNNModel_LiChan2014_AvgPool = cm.deepcopy(hp_default)
hp_CNNModel_LiChan2014_AvgPool['model_params']['vision_model_lib'] = 'CNNModel_LiChan2014_AvgPool'
hp_CNNModel_LiChan2014_AvgPool['model_params']['vision_model_path'] = './checkpoints/CNNModel_LiChan2014_AvgPool.hp_default.ckpt'
