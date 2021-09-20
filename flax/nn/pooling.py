# Copyright 2021 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pooling modules."""

from jax import lax
import jax.numpy as jnp

import numpy as np


def pool(inputs, init, reduce_fn, window_shape, strides, padding):
  """DEPRECATION WARNING:
  The `flax.nn` module is Deprecated, use `flax.linen` instead. 
  Learn more and find an upgrade guide at 
  https://github.com/google/flax/blob/main/flax/linen/README.md"
  Helper function to define pooling functions.

  Pooling functions are implemented using the ReduceWindow XLA op.
  NOTE: Be aware that pooling is not generally differentiable.
  That means providing a reduce_fn that is differentiable does not imply
  that pool is differentiable.

  Args:
    inputs: input data with dimensions (batch, window dims..., features).
    init: the initial value for the reduction
    reduce_fn: a reduce function of the form `(T, T) -> T`.
    window_shape: a shape tuple defining the window to reduce over.
    strides: a sequence of `n` integers, representing the inter-window
        strides.
    padding: either the string `'SAME'`, the string `'VALID'`, or a sequence
      of `n` `(low, high)` integer pairs that give the padding to apply before
      and after each spatial dimension.
  Returns:
    The output of the reduction for each window slice.
  """
  strides = strides or (1,) * len(window_shape)
  strides = (1,) + strides + (1,)
  dims = (1,) + window_shape + (1,)
  if not isinstance(padding, str):
    padding = tuple(map(tuple, padding))
    assert(len(padding) == len(window_shape)), (
      f"padding {padding} must specify pads for same number of dims as "
      f"window_shape {window_shape}")
    assert(all([len(x) == 2 for x in padding])), (
      f"each entry in padding {padding} must be length 2")
    padding = ((0,0),) + padding + ((0,0),)
  return lax.reduce_window(inputs, init, reduce_fn, dims, strides, padding)


def avg_pool(inputs, window_shape, strides=None, padding="VALID"):
  """DEPRECATION WARNING:
  The `flax.nn` module is Deprecated, use `flax.linen` instead. 
  Learn more and find an upgrade guide at 
  https://github.com/google/flax/blob/main/flax/linen/README.md"
  Pools the input by taking the average over a window.

  Args:
    inputs: input data with dimensions (batch, window dims..., features).
    window_shape: a shape tuple defining the window to reduce over.
    strides: a sequence of `n` integers, representing the inter-window
        strides (default: `(1, ..., 1)`).
    padding: either the string `'SAME'`, the string `'VALID'`, or a sequence
      of `n` `(low, high)` integer pairs that give the padding to apply before
      and after each spatial dimension (default: `'VALID'`).
  Returns:
    The average for each window slice.
  """
  y = pool(inputs, 0., lax.add, window_shape, strides, padding)
  y = y / np.prod(window_shape)
  return y


def max_pool(inputs, window_shape, strides=None, padding="VALID"):
  """DEPRECATION WARNING:
  The `flax.nn` module is Deprecated, use `flax.linen` instead. 
  Learn more and find an upgrade guide at 
  https://github.com/google/flax/blob/main/flax/linen/README.md"
  Pools the input by taking the maximum of a window slice.

  Args:
    inputs: input data with dimensions (batch, window dims..., features).
    window_shape: a shape tuple defining the window to reduce over.
    strides: a sequence of `n` integers, representing the inter-window
        strides (default: `(1, ..., 1)`).
    padding: either the string `'SAME'`, the string `'VALID'`, or a sequence
      of `n` `(low, high)` integer pairs that give the padding to apply before
      and after each spatial dimension (default: `'VALID'`).
  Returns:
    The maximum for each window slice.
  """
  y = pool(inputs, -jnp.inf, lax.max, window_shape, strides, padding)
  return y
