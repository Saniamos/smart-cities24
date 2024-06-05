import metrics.Maokai as maokai
import lightning as pl
import torch
from loss.CompositionalLoss import CompositionalLoss
from copy import deepcopy

"""
Common Functions and Patterns of NNs in this Task.
"""

class NeuralNetwork(pl.LightningModule):
    def __init__(self, model_params, optimizer, loss_function, optimizer_params, scheduler_params):
        super().__init__()
        self.model_params = model_params
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.optimizer_params = optimizer_params
        self.scheduler_params = scheduler_params

        self.save_hyperparameters()

    def forward(self, x):
        return self.layers(x)

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), **self.optimizer_params)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, **self.scheduler_params)
        return [optimizer], [scheduler]

    def _forward_ensure_precision(self, batch):
        # make sure the data is float64/double precision, so that we don't have under/overflow in the metric/loss calc
        x, y = batch
        return self(x).double(), y.double()

    def training_step(self, train_batch, batch_idx):
        logits, y = self._forward_ensure_precision(train_batch)
        loss = self.loss_function(logits, y)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        logits, y = self._forward_ensure_precision(val_batch)
        loss = self.loss_function(logits, y)
        self.log("val_loss", loss, sync_dist=True)
        return loss
    
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        y_hat, y = self._forward_ensure_precision(batch)

        # print(y_hat.shape, y.shape)
        metrics = dict(
            loss = self.loss_function(y_hat, y),
            pck = maokai.pck(y_hat, y),
            pmpkpe = maokai.pmpkpe(y_hat, y, has_rot_data = False),
            mpkpe = maokai.mpkpe(y_hat, y)
        )

        self.log_dict(metrics)
        return metrics
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # because i was to lazy to create another dataloader for predictions 
        return self._forward_ensure_precision(batch)



data_params = dict(
    shuffle = True,
    batch_size = 128,
    fix_nan = True,
)

model_params = dict(
    output_size = 63,
)

scheduler_params = dict(
    milestones = [10, 20],
    gamma = 0.1,
)

optimizer_params_adam = dict(
    lr = 1e-4,
)

optimizer_params_sgd = dict(
    lr = 0.03,
    weight_decay = 0.0002,
    momentum = 0.9,
)

trainer_params = dict(
    max_epochs = 100,
    precision = "32",
)

early_stopping_params = dict(
    monitor = "val_loss",
    mode = "min",
    patience = 5,
)


def adjust_for_compositional_loss(hp, use_sgd=True):
    hp = deepcopy(hp)
    hp['loss_function'] = CompositionalLoss()
    if use_sgd:
        hp['optimizer'] = torch.optim.SGD
        hp['optimizer_params'] = optimizer_params_sgd
    return hp