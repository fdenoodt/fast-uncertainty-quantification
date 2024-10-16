import os
from typing import Tuple, Optional
import lightning as L
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

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


class UQ:
    @staticmethod
    # Function to compute model's uncertainty (e.g., softmax variance)
    def compute_uncertainty(outputs):  # outputs: (batch_size, num_classes)
        # Softmax the output to get probabilities
        probs = F.softmax(outputs, dim=1)

        # Compute variance (uncertainty) across the class probabilities
        uncertainty = probs.var(dim=1)
        return uncertainty

    @staticmethod
    def _variances_vs_accuracy_per_input_img(classifier: Classifier, batch: Tuple[Tensor, Tensor]) -> Tensor:
        """Returns tensor of shape (batch, 2),
        where the first column is the variance and the second column
        is 1 or 0 if the prediction is correct or not."""

        x, y = batch
        assert x.dim() == 5  # (batch, nb_views, C, H, W)

        # 1) per view, compute the predictions
        predictions, _ = classifier.forward_multiple(x)  # (batch_size, num_views, num_classes)
        assert len(predictions.shape) == 3, f"Predictions shape is {predictions.shape}, expected 3 dimensions."

        # softmax the predictions
        predictions = F.softmax(predictions, dim=2)  # (batch_size, num_views, num_classes)

        # 2) compute the variance of the predictions
        variance = predictions.var(dim=1)  # (batch_size, num_classes)
        average_variance = variance.mean(dim=1)  # (batch_size)

        # 3) get mean of the predictions and take argmax
        final_predictions = predictions.mean(dim=1).argmax(dim=1)  # (batch_size)

        # 4) compute the accuracy of the mode with the labels
        accuracy = (final_predictions == y)  # (batch_size)

        # 5) stack the variance and accuracy
        stack = torch.stack((average_variance, accuracy), dim=1)  # (batch_size, 2)
        return stack

    @staticmethod
    def variances_vs_accuracy_per_input_img(classifier: Classifier, data_loader: DataLoader,
                                            limit_batches: float) -> Tensor:
        """Returns tensor of shape (batch, 2),
        where the first column is the variance and the second column
        is 1 or 0 if the prediction is correct or not."""

        assert 0 <= limit_batches <= 1, f"limit_batches must be between 0 and 1, got {limit_batches}."

        classifier.eval()
        var_vs_accuracy = None

        total_batches = len(data_loader)
        num_batches_to_process = int(total_batches * limit_batches)

        for i, batch in enumerate(data_loader):
            if i >= num_batches_to_process:
                break

            batch_var_vs_accuracy = UQ._variances_vs_accuracy_per_input_img(classifier, batch)
            if var_vs_accuracy is None:
                var_vs_accuracy = batch_var_vs_accuracy
            else:
                var_vs_accuracy = torch.cat((var_vs_accuracy, batch_var_vs_accuracy), dim=0)

        return var_vs_accuracy
