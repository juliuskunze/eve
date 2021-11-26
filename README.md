# Adaptive Optimization with Examplewise Gradients

Implementation of the Eve optimizer and experiments from [Adaptive Optimization with Examplewise Gradients](https://arxiv.org/pdf/2112.00174.pdf).

If you find this code useful, please reference in your paper:

```
@article{kunze2021eve,
title={Adaptive Optimization with Examplewise Gradients},
author={Kunze, Julius and Townsend, Jamie and Barber, David},
journal={arXiv preprint arXiv:2112.00174},
year={2021}
}
```

## Install

[Install JAX](https://github.com/google/jax#installation).

Install dependencies for experiments of choice:

```
pip install -r <cifar10|seq2seq|wmt|ppo|pixelcnn>/requirements.txt
```

## Run

See `README.md`s in subfolders to run experiments.
Default configs are as used in the paper.
Results are logged to [wandb](https://wandb.com).