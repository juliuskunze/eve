## Installation

You will need Python 3.6 or later.

For GPU support, first install `jaxlib`; please follow the
instructions in the [JAX
readme](https://github.com/google/jax/blob/main/README.md).  If they
are not already installed, you will need to install
[CUDA](https://developer.nvidia.com/cuda-downloads) and
[CuDNN](https://developer.nvidia.com/cudnn) runtimes.

Then install `flax` from PyPi:

```
> pip install flax
```

To upgrade to the latest version of JAX and Flax, you can use:

```
> pip install --upgrade pip jax jaxlib
> pip install --upgrade git+https://github.com/google/flax.git
```
