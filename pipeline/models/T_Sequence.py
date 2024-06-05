import torch
from torch import nn
import models.common as cm
import importlib
import metrics.Maokai as maokai

class NeuralNetwork(cm.NeuralNetwork):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.lstm = nn.LSTM(input_size=self.model_params["input_size"], 
                            # proj_size=self.model_params["output_size"],
                            hidden_size=self.model_params["hidden_size"], 
                            # num_layers=self.model_params["num_layers"], 
                            # bidirectional=False,
                            batch_first=True)
        
        self.fc = nn.Linear(self.model_params["hidden_size"], self.model_params["output_size"])
        
    # input dimensions are: batch, seq, skeleton
    def forward(self, data):
        # return vision_out # for testing -> this works, the results are the same as in single training -> reshaping is not the issue here
        # in the lstm we pass batch, seq, skeleton (the latter is the output of the cnn)
        # print(x_3d.shape, vision_out.shape)
        # h0 = torch.ones(1, x_3d.shape[0], self.model_params['output_size'], device=vision_out.device)
        # h0 = torch.zeros(1, x_3d.shape[0], self.model_params['output_size'], device=vision_out.device)
        # c0 = torch.ones(1, x_3d.shape[0], self.model_params['hidden_size'], device=vision_out.device)
        # out, _ = self.lstm(vision_out, h0)
        # out, _ = self.lstm(vision_out, (h0, c0))
        # out, _ = self.lstm(out)
        _, (hn, _) = self.lstm(data)
        # print(x_3d.shape, out.shape, hn.shape)
        out = hn.view(-1, self.model_params["hidden_size"])
        out = self.fc(out)
        # out = self.fc(vision_out)
        return out
    

hp_default = dict(
    loss_function = nn.MSELoss(),
    optimizer = torch.optim.Adam,

    data_params = dict(
        batch_size= 512,
        context= 15, 
        shuffle= True,

        # file = 'CNNModel_LiChan2014.hp_default'
        file = 'ResNet50_SunShangetAl.hp_l1_loss'
    ),

    model_params = dict(
        input_size = 63,
        output_size = 63,
        hidden_size = 1024,
        num_layers = 1,
    ),

    scheduler_params = cm.scheduler_params,
    optimizer_params = cm.optimizer_params_adam,
    trainer_params = dict(
        max_epochs = 300,
        precision = "32",
    ),
    early_stopping_params = cm.early_stopping_params,
    
)

hp_default_OwnBaselineCNN = cm.deepcopy(hp_default)
hp_default_OwnBaselineCNN['data_params']['file'] = 'OwnBaselineCNN.hp_default'

hp_default_LiChan2014 = cm.deepcopy(hp_default)
hp_default_LiChan2014['data_params']['file'] = 'CNNModel_LiChan2014.hp_default'

hp_l1_loss = cm.deepcopy(hp_default)
hp_l1_loss['loss_function'] = nn.L1Loss()
hp_l1_loss['data_params']['batch_size'] = 256

hp_l1_loss_b256 = cm.deepcopy(hp_default)
hp_l1_loss_b256['loss_function'] = nn.L1Loss()
hp_l1_loss_b256['data_params']['batch_size'] = 256
hp_l1_loss_b256['trainer_params']['max_epochs'] = 100

hp_l1_loss_b2048 = cm.deepcopy(hp_default)
hp_l1_loss_b2048['loss_function'] = nn.L1Loss()
hp_l1_loss_b2048['data_params']['batch_size'] = 2048
hp_l1_loss_b2048['trainer_params']['max_epochs'] = 100


hp_l1_loss_600 = cm.deepcopy(hp_default)
hp_l1_loss_600['loss_function'] = nn.L1Loss()
hp_l1_loss_600['trainer_params']['max_epochs'] = 600


hp_l1_loss_ctx_60 = cm.deepcopy(hp_default)
hp_l1_loss_ctx_60['loss_function'] = nn.L1Loss()
hp_l1_loss_ctx_60['data_params']['context'] = 60
# hp_l1_loss_ctx_60['trainer_params']['max_epochs'] = 500

hp_l1_loss_ctx_60_b256 = cm.deepcopy(hp_default)
hp_l1_loss_ctx_60_b256['loss_function'] = nn.L1Loss()
hp_l1_loss_ctx_60_b256['data_params']['batch_size'] = 256
hp_l1_loss_ctx_60_b256['data_params']['context'] = 60


hp_dropna = cm.deepcopy(hp_default)
hp_dropna['loss_function'] = nn.L1Loss()
hp_dropna['data_params']['file'] = 'ResNet50_SunShangetAl.hp_dropna'

hp_dropna_all = cm.deepcopy(hp_default)
hp_dropna_all['loss_function'] = nn.L1Loss()
hp_dropna_all['data_params']['file'] = 'ResNet50_SunShangetAl.hp_dropna.all'