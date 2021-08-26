# encoding: utf-8

# last update: xiaoya li
# issue: https://github.com/PyTorchLightning/pytorch-lightning/issues/1868
# set for trainer: https://pytorch-lightning.readthedocs.io/en/latest/trainer.html
#   from pytorch_lightning import Trainer, seed_everything
#   seed_everything(42)
#   sets seeds for numpy, torch, python.random and PYTHONHASHSEED.
#   model = Model()
#   trainer = Trainer(deterministic=True)

import random
import paddle
import numpy as np

def set_random_seed(seed: int):
    """set seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


if __name__ == '__main__':
    # without this line, x would be different in every execution.
    set_random_seed(0)

    x = np.random.random()
    print(x)
