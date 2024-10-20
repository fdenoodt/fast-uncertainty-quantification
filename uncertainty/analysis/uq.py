from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from shared.data import filter_iterator
from shared.data.transformations import Noise
from uncertainty.uq_through_redundancy.model import Classifier
import numpy as np


class UQ:  # Uncertainty Quantification
    @staticmethod
    # Function to compute model's uncertainty (e.g., softmax variance)
    def compute_entropy(outputs):
        """
        Returns entropy(probabilities_averaged_across_views).
        in: outputs: (batch_size, nb_views, num_classes)
        out: uncertainty: (batch_size)
        """
        # outputs: (batch_size, nb_views, num_classes)
        # Softmax the output to get probabilities
        probs = F.softmax(outputs, dim=1)  # (batch_size, nb_views, num_classes)

        # Compute variance (uncertainty) across the class probabilities
        # uncertainty = probs.var(dim=1) # (batch_size, num_classes)

        # Compute entropy across the class probabilities
        mean_probs = probs.mean(dim=1)  # (batch_size, num_classes)
        uncertainty = -(mean_probs * torch.log(mean_probs)).sum(dim=1)  # (batch_size)

        return uncertainty  # (batch_size)

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

        for i, batch in filter_iterator(data_loader, limit_batches):
            batch_var_vs_accuracy = UQ._variances_vs_accuracy_per_input_img(classifier, batch)
            if var_vs_accuracy is None:
                var_vs_accuracy = batch_var_vs_accuracy
            else:
                var_vs_accuracy = torch.cat((var_vs_accuracy, batch_var_vs_accuracy), dim=0)

        return var_vs_accuracy

    @staticmethod
    def retrieve_entropy_data(model, test_loader, limit_batches: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Iterates over different noise levels and computes the mean entropy of the model's predictions.
        :param model:
        :param test_loader:
        :param limit_batches:
        :return:
        """
        assert 0 <= limit_batches <= 1, f"limit_batches must be between 0 and 1, got {limit_batches}."

        model.eval()

        # List to store noise levels and corresponding uncertainty values
        noise_levels: np.ndarray = np.linspace(0, 1, 10)
        mean_uncertainties = []

        for noise_factor in noise_levels:
            all_uncertainties = []

            for i, (images, _) in filter_iterator(test_loader, limit_batches):
                noisy_images = Noise.add_gaussian_noise(images, noise_factor)
                with torch.no_grad():
                    outputs, _ = model(noisy_images)
                    entropies = UQ.compute_entropy(outputs)  # (batch_size)
                    # to list and append all
                    all_uncertainties.extend(entropies.tolist())

            mean_uncertainties.append(np.mean(all_uncertainties))  # scalar

        return noise_levels, np.array(mean_uncertainties)
