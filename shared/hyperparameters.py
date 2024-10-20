import argparse
import sys


class Hyperparameters:
    @staticmethod
    def get():
        h = {  # hyperparameters
            'method': 'uq_through_redundancy',  # uq_through_redundancy, ensemble
            'redundancy_method': 'aug_rotate',  # augmentations, splitting into chunks, etc.

            'dataset': 'MNIST',  # 'TOY_REGRESSION', 'RADIO',
            'dataset_path': '../data/',

            'num_views': 5,
            # 'loss': 'cross_entropy',  # 'cross_entropy', 'cross_entropy_with_distinct_path_regularization',
            'alpha': 1,
            # 0.35 seems to kinda work w/ acc 0.9,  # 1 for cross_entropy, 0 for distinct_path_regularization

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

            'use_wandb': True,

            'wandb_project': 'uncertainty',
            'wandb_entity': 'oBoii',
            'wandb_project_name': 'test',

            'train': True,  # Set this to false if you only want to evaluate the model
            'fast_dev_run': False,
            'overfit_batches': 0.0
        }
        h['checkpoint_path'] = f"{h['log_path']}/saved_models"
        h = Hyperparameters.overwrite_args_cli(h)

        Hyperparameters.apply_sanity_checks(h)
        return h

    @staticmethod
    def apply_sanity_checks(h):
        pass
        # assert h['loss'] in ['cross_entropy', 'cross_entropy_with_distinct_path_regularization']

    @staticmethod
    def overwrite_args_cli(h: dict[str, any]) -> dict[str, any]:
        # Check if the script is being run in a Jupyter notebook
        if 'ipykernel' not in sys.modules:
            # Parse command-line arguments
            parser = argparse.ArgumentParser()
            for key, value in h.items():
                if isinstance(value, bool):
                    parser.add_argument(f'--{key}', type=bool, default=value)
                elif isinstance(value, int):
                    parser.add_argument(f'--{key}', type=int, default=value)
                elif isinstance(value, float):
                    parser.add_argument(f'--{key}', type=float, default=value)
                else:  # for str and potentially other types
                    parser.add_argument(f'--{key}', type=type(value), default=value)
            args = parser.parse_args()

            # Overwrite the default hyperparameters with the command-line arguments
            h.update(vars(args))

        # e.g.: python main.py --method uq_through_redundancy --dataset MNIST --num_views 5 --alpha 0.1 --train True
        return h
