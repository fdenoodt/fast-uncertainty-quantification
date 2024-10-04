import argparse
import sys


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

    return h
