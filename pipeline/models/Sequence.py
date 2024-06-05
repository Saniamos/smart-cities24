import torch
from torch import nn
import models.common as cm
import importlib

"""
Genauso wie CNNModel_LiChan2014 nur mit AveragePooling Layer

CNN aus Paper: 3D Human Pose Estimation from Monocular Images with Deep Convolutional Neural Network.
Average Pooling als ersten Layern hinzugefügt.
"""

name = "SequenceModel"

class NeuralNetwork(cm.NeuralNetwork):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        VisionModule = importlib.import_module(f"models.{self.model_params['vision_model_lib']}")
        self.vision_model = VisionModule.NeuralNetwork.load_from_checkpoint(self.model_params['vision_model_path'])

        if not self.model_params['retrain']:
            # Freeze all layers
            for param in self.vision_model.parameters():
                param.requires_grad = False
            
            # Unfreeze the last layer
            for param in self.vision_model.layers[-1].parameters():
                param.requires_grad = True

        self.lstm = nn.LSTM(input_size=self.model_params["output_size"], 
                            # proj_size=self.model_params["output_size"],
                            hidden_size=self.model_params["hidden_size"], 
                            # num_layers=self.model_params["num_layers"], 
                            # bidirectional=False,
                            batch_first=True)
        
        self.fc = nn.Linear(self.model_params["hidden_size"], self.model_params["output_size"])
        
    def _apply_vision(self, x_3d):
        batch, seq, pix, height, width = x_3d.shape
        # i've tested this without the lstm and it reproduces the same predictions as without the lstm -> the reshaping is correct here
        return self.vision_model(x_3d.view(batch * seq, pix, height, width)).view(batch, seq, -1)

    # input dimensions are: batch, seq, pixel, height, width
    def forward(self, x_3d):
        out = self._apply_vision(x_3d)
        # return vision_out # for testing -> this works, the results are the same as in single training -> reshaping is not the issue here
        # in the lstm we pass batch, seq, skeleton (the latter is the output of the cnn)
        # print(x_3d.shape, vision_out.shape)
        # h0 = torch.ones(1, x_3d.shape[0], self.model_params['output_size'], device=vision_out.device)
        # h0 = torch.zeros(1, x_3d.shape[0], self.model_params['output_size'], device=vision_out.device)
        # c0 = torch.ones(1, x_3d.shape[0], self.model_params['hidden_size'], device=vision_out.device)
        # out, _ = self.lstm(vision_out, h0)
        # out, _ = self.lstm(vision_out, (h0, c0))
        # out, _ = self.lstm(out)
        _, (hn, _) = self.lstm(out)
        # print(x_3d.shape, out.shape, hn.shape)
        out = hn.view(-1, self.model_params["hidden_size"])
        out = self.fc(out)
        # out = self.fc(vision_out)
        return out
    
    
hp_default = dict(
    loss_function = nn.MSELoss(),
    optimizer = torch.optim.Adam,

    data_params = {**cm.data_params, 
                   'batch_size': 192,
                   'context': 15, # 30hz * 1s
                   'shuffle': True,
                   },

    model_params = dict(
        vision_model_lib = 'CNNModel_LiChan2014_AvgPool',
        vision_model_path = './checkpoints/CNNModel_LiChan2014_AvgPool.hp_default.ckpt',
        output_size = 63,
        hidden_size = 1024,
        retrain = False,
        num_layers = 1,
    ),

    scheduler_params = cm.scheduler_params,
    optimizer_params = cm.optimizer_params_adam,
    trainer_params = cm.trainer_params,
    early_stopping_params = cm.early_stopping_params,
    
)
