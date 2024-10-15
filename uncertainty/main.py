import matplotlib.pyplot as plt
import torch
from lightning import Trainer
from torchvision import transforms
from torchvision.datasets import MNIST
from shared import overwrite_args_cli
from shared.data.data_loader import data_loaders
from shared.loss import Loss
from shared.decorators import init_decorator, wandb_decorator, timer_decorator
from shared.data.transformations import T, Noise
from uncertainty.uq_through_redundancy.model import Classifier, UQ

from uncertainty.analysis import Visual


@init_decorator
@wandb_decorator  # calls wandb.init
@timer_decorator
def main(h: dict):
    train_loader, val_loader = data_loaders(
        h,
        [lambda _: T.create_multiple_views(_, h['num_views'], T.subsample_with_random_transform)])

    c = Classifier(
        h=h,
        in_channels=h['in_channels'],
        latent_dims=h['latent_dims'],
        latent_strides=h['latent_strides'],
        latent_kernel_sizes=h['latent_kernel_sizes'],
        num_classes=h['num_classes'],
        loss=Loss(h['alpha'])
    )

    trainer = Trainer(max_epochs=h['epochs'], fast_dev_run=h['fast_dev_run'], overfit_batches=h['overfit_batches'],
                      devices="auto", strategy="auto",
                      limit_train_batches=h['limit_train_batches'],
                      limit_val_batches=h['limit_val_batches'],
                      limit_test_batches=h['limit_test_batches'])

    if h['train']:
        try:
            trainer.fit(c, train_loader, val_loader)
        except KeyboardInterrupt:
            print("Training interrupted. Saving model...")
        finally:
            c.save("model0.pth")

    # Evaluate the model
    results = trainer.test(c, val_loader)

    # Print accuracy
    print(f"Test Accuracy: {results[0]['test_acc']:.4f}")
    return c


if __name__ == "__main__":
    h = {  # hyperparameters
        'method': 'uq_through_redundancy',  # uq_through_redundancy, ensemble
        'dataset': 'MNIST',  # 'TOY_REGRESSION', 'RADIO',
        'dataset_path': '../data/',

        'num_views': 5,
        # 'loss': 'cross_entropy',  # 'cross_entropy', 'cross_entropy_with_distinct_path_regularization',
        'alpha': 1,  # 0.35 seems to kinda work w/ acc 0.9,  # 1 for cross_entropy, 0 for distinct_path_regularization

        'in_channels': 1,
        'latent_dims': [32, 64],
        'latent_strides': [2, 2],
        'latent_kernel_sizes': [3, 3],
        'num_classes': 10,

        'log_path': './logs/',
        'epochs': 2,
        'learning_rate': 1e-3,
        'batch_size': 32,
        'num_workers': 4,

        'limit_train_batches': 1.0,
        'limit_val_batches': 1.0,
        'limit_test_batches': 1.0,

        'seed': 49,

        'use_wandb': False,

        'wandb_project': 'uncertainty',
        'wandb_entity': 'oBoii',
        'wandb_name': 'radio',

        'train': True,  # Set this to false if you only want to evaluate the model
        'fast_dev_run': True,
        'overfit_batches': 0.0
    }

    h['checkpoint_path'] = f"{h['log_path']}/saved_models"

    h = overwrite_args_cli(h)

    # assert h['loss'] in ['cross_entropy', 'cross_entropy_with_distinct_path_regularization']

    Visual.show_transformations(h)

    c = main(h)

    c.eval()
    train, val = data_loaders(
        h, [lambda _: T.create_multiple_views(_, h['num_views'], T.subsample_with_random_transform)])

    # (batch_size, 2) where 1st column is variance and 2nd is 1 or 0 if the prediction is correct or not
    res = UQ.variances_vs_accuracy_per_input_img(c, val)

    Visual.DiscrDistr.variances_per_wrong_or_correct_pred(res.detach())
    Visual.DiscrDistr.variances_per_wrong_or_correct_pred_normalized(res.detach())
    Visual.ContinDistr.variances_per_wrong_or_correct_pred(res.detach())
    Visual.plot_uncertainty_vs_noise_level(c, val)
