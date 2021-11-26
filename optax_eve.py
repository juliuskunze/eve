from typing import Callable, Any, Union, Optional, Tuple

import chex
import jax
from jax import numpy as jnp, lax
from optax._src import combine, transform, base, numerics
from optax._src.alias import ScalarOrSchedule, _scale_by_learning_rate
from optax._src.transform import _bias_correction


class ScaleByEveState(base.OptState):
  """State for the Adam algorithm."""
  count: chex.Array  # shape=(), dtype=jnp.int32.
  mu: base.Updates
  nu: base.Updates


def _update_elementwise_moment(updates, moments, decay, order, batch_axes):
  return jax.tree_multimap(
    lambda g, t: (1 - decay) * (g ** order).mean(batch_axes) + decay * t, updates, moments)


def scale_by_eve(
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    batch_axes: Tuple[Union[int, str]] = (0,),
) -> base.GradientTransformation:
  def init_fn(params):
    mu = jax.tree_map(jnp.zeros_like, params)  # First moment
    nu = jax.tree_map(jnp.zeros_like, params)  # Second moment
    return ScaleByEveState(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu)

  def update_fn(updates, state, params=None):
    del params
    mu = _update_elementwise_moment(updates, state.mu, b1, 1, batch_axes)
    nu = _update_elementwise_moment(updates, state.nu, b2, 2, batch_axes)
    count_inc = numerics.safe_int32_increment(state.count)
    mu_hat = _bias_correction(mu, b1, count_inc)
    nu_hat = _bias_correction(nu, b2, count_inc)
    b = 1 / jax.tree_leaves(updates)[0].shape[0]
    updates = jax.tree_multimap(
      lambda m, v: m / (jnp.sqrt(v * b + lax.square(m) * (1 - b) + eps_root) + eps), mu_hat, nu_hat)
    return updates, ScaleByEveState(count=count_inc, mu=mu, nu=nu)

  return base.GradientTransformation(init_fn, update_fn)


def eve(
    learning_rate: ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    batch_axes: Tuple[Union[int, str]] = (0,),
) -> base.GradientTransformation:
  return combine.chain(
    scale_by_eve(b1=b1, b2=b2, eps=eps, eps_root=eps_root, batch_axes=batch_axes),
    _scale_by_learning_rate(learning_rate))


def evew(
    learning_rate: ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    batch_axes: Tuple[Union[int, str]] = (0,),
    weight_decay: float = 1e-4,
    mask: Optional[Union[Any, Callable[[base.Params], Any]]] = None,
) -> base.GradientTransformation:
  return combine.chain(
    scale_by_eve(b1=b1, b2=b2, eps=eps, eps_root=eps_root, batch_axes=batch_axes),
    transform.add_decayed_weights(weight_decay, mask),
    _scale_by_learning_rate(learning_rate),
  )
