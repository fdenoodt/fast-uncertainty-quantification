from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger

from shared.data.data_loader import data_loaders
from shared.data.transformations import T
from shared.decorators import init_decorator, wandb_decorator, timer_decorator
from shared.hyperparameters import Hyperparameters
from shared.loss import Loss
from shared.wandb import W  # wandb wrapper

from uncertainty.analysis import Visual
from uncertainty.uq_through_redundancy.model import Classifier, UQ


def train(h: dict, c: Classifier, trainer: Trainer, train_loader, val_loader):
    if h['train']:
        try:
            trainer.fit(c, train_loader, val_loader)
        except KeyboardInterrupt:
            print("Training interrupted. Saving model...")
        finally:
            c.save("model0.pth")


def analysis(h: dict, c: Classifier, trainer: Trainer, val_loader):
    # Evaluate the model
    results = trainer.test(c, val_loader)

    # Print accuracy
    print(f"Test Accuracy: {results[0]['test_acc']:.4f}")

    c.eval()
    train, val = data_loaders(
        h, [lambda _: T.create_multiple_views(_, h['num_views'], T.retrieve_transforms(h))])

    # (batch_size, 2) where 1st column is variance and 2nd is 1 or 0 if the prediction is correct or not
    res = UQ.variances_vs_accuracy_per_input_img(c, val, limit_batches=h['limit_val_batches'])

    W.log_im(
        h,
        Visual.DiscrDistr.correct_vs_wrong(res.detach()),
        "DiscrDistr")
    W.log_im(
        h,
        Visual.ContinDistr.correct_vs_wrong(res.detach()),
        "ContinDistr")

    W.log_im(
        h,
        Visual.DiscrDistr.correct_vs_wrong_normalized(res.detach()),
        "DiscrDistr_normalized")

    W.log_im(
        h,
        Visual.plot_uncertainty_vs_noise_level(c, val, limit_batches=h['limit_val_batches']),
        "Uncertainty_vs_noise_level")


@init_decorator
@wandb_decorator  # calls wandb.init
@timer_decorator
def main(h: dict):
    train_loader, val_loader = data_loaders(
        h,
        [lambda _: T.create_multiple_views(_, h['num_views'], T.retrieve_transforms(h))])

    c = Classifier(
        h=h,  # Hyperparameters
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
                      limit_test_batches=h['limit_test_batches'],
                      logger=WandbLogger() if h['use_wandb'] else None)

    train(h, c, trainer, train_loader, val_loader)
    analysis(h, c, trainer, val_loader)
    return c


if __name__ == "__main__":
    h = Hyperparameters.get()
    Visual.show_transformations(h)

    c = main(h)
