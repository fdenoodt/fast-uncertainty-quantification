import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch import Tensor
from torchvision import transforms
from torchvision.datasets import MNIST

from shared.data.transformations import Noise, T
from uncertainty.uq_through_redundancy.model import UQ


class Visual:
    class DiscrDistr:
        @staticmethod
        def variances_per_wrong_or_correct_pred(var_vs_accuracy: Tensor):
            variances = var_vs_accuracy[:, 0].numpy()  # scalar values
            accuracies = var_vs_accuracy[:, 1].numpy()  # 0 or 1s

            fig, ax = plt.subplots()
            colors = ['blue', 'red']  # Colors for correct and wrong predictions
            labels = ['Correct Predictions', 'Wrong Predictions']

            for i, use_correct_predictions in enumerate([1., 0.]):
                indices = np.where(accuracies == use_correct_predictions)[0]
                variances_filtered = variances[indices]
                ax.hist(variances_filtered, bins=25, color=colors[i], alpha=0.5, label=labels[i])

            ax.set_xlabel("Variance")
            ax.set_ylabel("Count")
            ax.set_title("Variance Distribution for Correct vs Wrong Predictions")
            ax.legend()

            # Log the matplotlib figure to wandb
            # if options.use_wandb:
            #     wandb_section = get_wandb_audio_classific_key(opt, classifier_config)
            #     wandb.log({f"{wandb_section}_softmax/variance_distribution_combined": wandb.Image(fig)})

            plt.show()
            plt.close(fig)

        @staticmethod
        def variances_per_wrong_or_correct_pred_normalized(var_vs_accuracy: Tensor):
            variances = var_vs_accuracy[:, 0].numpy()  # scalar values
            accuracies = var_vs_accuracy[:, 1].numpy()  # 0 or 1s

            fig, ax = plt.subplots()
            colors = ['blue', 'red']  # Colors for correct and wrong predictions
            labels = ['Correct Predictions', 'Wrong Predictions']

            for i, use_correct_predictions in enumerate([1., 0.]):
                indices = np.where(accuracies == use_correct_predictions)[0]
                variances_filtered = variances[indices]
                ax.hist(variances_filtered, bins=25, color=colors[i], alpha=0.5, label=labels[i], density=True)

            ax.set_xlabel("Variance")
            ax.set_ylabel("Density")
            ax.set_title("Variance Distribution for Correct vs Wrong Predictions")
            ax.legend()

            # Log the matplotlib figure to wandb
            # if options.use_wandb:
            #     wandb_section = get_wandb_audio_classific_key(opt, classifier_config)
            #     wandb.log({f"{wandb_section}_softmax/variance_distribution_combined": wandb.Image(fig)})

            plt.show()
            plt.close(fig)

    class ContinDistr:
        @staticmethod
        def variances_per_wrong_or_correct_pred(var_vs_accuracy: Tensor):
            variances = var_vs_accuracy[:, 0].numpy()  # scalar values
            accuracies = var_vs_accuracy[:, 1].numpy()  # 0 or 1s

            fig, ax = plt.subplots()
            colors = ['blue', 'red']  # Colors for correct and wrong predictions
            labels = ['Correct Predictions', 'Wrong Predictions']

            for i, use_correct_predictions in enumerate([1., 0.]):
                indices = np.where(accuracies == use_correct_predictions)[0]
                variances_filtered = variances[indices]
                sns.kdeplot(variances_filtered, color=colors[i], label=labels[i])

            ax.set_xlabel("Variance")
            ax.set_ylabel("Density")
            ax.set_title("Variance Distribution for Correct vs Wrong Predictions")
            ax.legend()

            # Log the matplotlib figure to wandb
            # if options.use_wandb:
            #     wandb_section = get_wandb_audio_classific_key(opt, classifier_config)
            #     wandb.log({f"{wandb_section}_softmax/variance_distribution_combined": wandb.Image(fig)})

            plt.show()
            plt.close(fig)

    @staticmethod
    def plot_uncertainty_vs_noise_level(model, test_loader):
        model.eval()

        # List to store noise levels and corresponding uncertainty values
        noise_levels = np.linspace(0, 1, 10)
        mean_uncertainties = []

        for noise_factor in noise_levels:
            all_uncertainties = []

            for images, _ in test_loader:
                noisy_images = Noise.add_gaussian_noise(images, noise_factor)
                with torch.no_grad():
                    outputs, _ = model(noisy_images)
                    uncertainty = UQ.compute_uncertainty(outputs)
                    all_uncertainties.append(uncertainty.mean().item())

            mean_uncertainties.append(np.mean(all_uncertainties))

        # Plotting the results
        plt.figure(figsize=(8, 6))
        plt.plot(noise_levels, mean_uncertainties, marker='o', linestyle='-')
        plt.title('Model Uncertainty under Distributional Shift (Gaussian Noise)')
        plt.xlabel('Noise Level (std dev of Gaussian noise)')
        plt.ylabel('Mean Uncertainty (Softmax Variance)')
        plt.grid(True)
        plt.show()

    @staticmethod
    def show_transformations(h):
        # dispay the different subsampling methods, same image repeated 3 times
        transform = transforms.Compose([transforms.ToTensor()])  # rnd_subsample_method])
        mnist_train = MNIST(h['dataset_path'], train=True, download=True, transform=transform)

        methods = [
            # subsample,
            #        subsample_with_random_stride,
            #        subsample_with_patch_dropout,
            T.subsample_with_random_transform,
            # subsample_with_pixel_swapping
        ]
        x, y = mnist_train[0]
        for method in methods:
            # create 3 images and display them side by side
            imgs = [method(x) for _ in range(3)]
            imgs = torch.cat(imgs, dim=2)
            plt.imshow(imgs.squeeze(0), cmap='gray')
            plt.title(method.__name__)
            plt.show()
