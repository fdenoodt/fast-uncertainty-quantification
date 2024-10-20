import os
from typing import Tuple, Optional
import lightning as L
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from shared.data import filter_iterator
from shared.loss import Loss


class Classifier(L.LightningModule):
    def __init__(self, h: dict, in_channels, latent_dims: list, latent_strides: list, latent_kernel_sizes: list,
                 num_classes: int, loss: Loss):
        super(Classifier, self).__init__()

        assert callable(loss), f"Loss must be a callable function, got {loss}."

        self.h = h
        self.latent_dims = latent_dims
        self.loss: Loss = loss

        # Encoder layers. Each layer will be a reparameterization layer
        layers = []
        for dim, stride, kernel_size in zip(latent_dims, latent_strides, latent_kernel_sizes):
            layers.append(nn.Conv2d(in_channels, dim, kernel_size, stride))
            layers.append(nn.BatchNorm2d(dim))
            in_channels = dim

        self.layers = nn.ModuleList(layers)
        self.fc = nn.Linear(latent_dims[-1], num_classes)

    def forward(self, x: Tensor):
        """Out: (batch, nb_views, num_classes), latent_representations
        In: (batch, nb_views, C, H, W) """
        assert x.dim() == 5, f"Input shape is {x.shape}, expected 5 dimensions. (batch, nb_views, C, H, W)"
        return self.forward_multiple(x)

    def forward_single(self, x) -> Tuple[Tensor, list]:
        """
        In: (batch, C, H, W)
        Out: (batch, num_classes), latent_representations
        """
        # (batch, C, H, W)
        assert x.dim() == 4, f"Input shape is {x.shape}, expected 4 dimensions."
        latent_representations = []
        for layer in self.layers:
            x = layer(x)
            if isinstance(layer, nn.BatchNorm2d):
                x = nn.functional.relu(x)
            latent_representations.append(x)

        # average pooling
        x = nn.functional.avg_pool2d(x, x.size()[2:]).view(x.size(0), -1)
        x = self.fc(x)
        return x, latent_representations

    def forward_multiple(self, x) -> Tuple[Tensor, list]:
        """Returns the predictions and latent representations of all views of the input.
        In: (batch, nb_views, C, H, W)
        Out: (batch, nb_views, num_classes), latent_representations
        """

        # (batch, nb_views, C, H, W)
        assert x.dim() == 5, f"Input shape is {x.shape}, expected 5 dimensions."

        batch, nb_views, c, h, w = x.shape
        x = x.view(-1, c, h, w)  # (batch * nb_views, C, H, W)
        x, latent_representations = self.forward_single(x)
        x = x.view(batch, nb_views, -1)  # (batch, nb_views, num_classes)

        # Reshape latent representations
        latent_representations = [l.view(batch, nb_views, *l.shape[1:]) for l in latent_representations]
        return x, latent_representations

    def training_step(self, batch, batch_idx):
        x, y = batch  # x: (batch, nb_views, C, H, W), y: (batch)
        y_hat, latent_representations = self(x)  # (batch, nb_views, num_classes)

        loss = self.loss(y_hat, y, latent_representations, x.size(1))
        self.log('train_loss', loss, batch_size=x.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat, latent_representations = self(x)

        loss = self.loss(y_hat, y, latent_representations, x.size(1))
        self.log('val_loss', loss, batch_size=x.size(0))
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch  # x: (batch, nb_views, C, H, W), y: (batch)
        y_hat, latent_representations = self(x)  # y_hat: (batch, nb_views, num_classes)

        loss = self.loss(y_hat, y, latent_representations, x.size(1))

        y_hat_mean = y_hat.mean(dim=1)
        acc = (y_hat_mean.argmax(dim=1) == y).float().mean()

        self.log('test_acc', acc)
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.h['learning_rate'])

    def save(self, name: Optional[str] = 'model.pth') -> None:
        os.makedirs(self.h['checkpoint_path'], exist_ok=True)
        torch.save(self.state_dict(), os.path.join(self.h['checkpoint_path'], name))

    def load(self) -> 'SimpleNet':
        self.load_state_dict(torch.load(os.path.join(self.h['checkpoint_path'], 'model.pth')))
        self.eval()
        return self
