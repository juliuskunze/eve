Learning Rate Scheduling
=============================
The learning rate is considered one of the most important hyperparameters for
training deep neural networks, but choosing it can be quite hard.
To simplify this, one can use a so-called *cyclic learning rate*, which
virtually eliminates the need for experimentally finding the best value and
schedule for the global learning rate. Instead of monotonically decreasing the
learning rate, this method lets the learning rate cyclically vary between
reasonable boundary values.
Here we will show you how to implement a triangular learning rate scheduler,
as described in the paper  `"Cyclical Learning Rates for Training Neural Networks" <https://arxiv.org/abs/1506.01186>`_.

We will show you how to...

* define a learning rate schedule
* train a simple model using that schedule

The triangular schedule makes your learning rate vary as a triangle wave during training, so over the course of a period (``steps_per_cycle``
training steps) the value will start at ``lr_min``, increase linearly to ``lr_max``, and then decrease again to ``lr_min``.

.. testsetup::

  import jax

.. testcode::
  
  def create_triangular_schedule(lr_min, lr_max, steps_per_cycle):
    top = (steps_per_cycle + 1) // 2
    def learning_rate_fn(step):
      cycle_step = step % steps_per_cycle
      if cycle_step < top:
        lr = lr_min + cycle_step/top * (lr_max - lr_min)
      else:
        lr = lr_max - ((cycle_step - top)/top) * (lr_max - lr_min)
      return lr
    return learning_rate_fn


To use the schedule, one must create a learning rate function by passing the hyperparameters to the
create_triangular_schedule function and then use that function to compute the learning rate for your updates.
For example using this schedule on MNIST would require changing the train_step function

.. codediff:: 
  :title_left: Default learning rate
  :title_right: Triangular learning rate schedule
  
  @jax.jit
  def train_step(optimizer, batch): #!
    def loss_fn(params):
      logits = CNN().apply({'params': params}, batch['image'])
      loss = cross_entropy_loss(logits, batch['label'])
      return loss, logits
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, logits), grad = grad_fn(optimizer.target)


    optimizer = optimizer.apply_gradient(grad) #!
    metrics = compute_metrics(logits, batch['label'])
    return optimizer, metrics
  ---
  @jax.jit
  def train_step(optimizer, batch, learning_rate_fn): #!
    def loss_fn(params):
      logits = CNN().apply({'params': params}, batch['image'])
      loss = cross_entropy_loss(logits, batch['label'])
      return loss, logits
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, logits), grad = grad_fn(optimizer.target)
    step = optimizer.state.step #!
    lr = learning_rate_fn(step) #!
    optimizer = optimizer.apply_gradient(grad, {"learning_rate": lr}) #!
    metrics = compute_metrics(logits, batch['label'])
    return optimizer, metrics

And the train_epoch function:

.. codediff::
  :title_left: Default learning rate
  :title_right: Triangular learning rate schedule
  
  def train_epoch(optimizer, train_ds, batch_size, epoch, rng):
  """Train for a single epoch."""
  train_ds_size = len(train_ds['image'])
  steps_per_epoch = train_ds_size // batch_size



  perms = jax.m random.permutation(rng, len(train_ds['image']))
  perms = perms[:steps_per_epoch * batch_size]
  perms = perms.reshape((steps_per_epoch, batch_size))
  batch_metrics = []
  for perm in perms:
    batch = {k: v[perm, ...] for k, v in train_ds.items()}
    optimizer, metrics = train_step(optimizer, batch) #!
    batch_metrics.append(metrics)

  # compute mean of metrics across each batch in epoch.
  batch_metrics = jax.device_get(batch_metrics)
  epoch_metrics = {
      k: np.mean([metrics[k] for metrics in batch_metrics])
      for k in batch_metrics[0]}

  logging.info('train epoch: %d, loss: %.4f, accuracy: %.2f', epoch,
               epoch_metrics['loss'], epoch_metrics['accuracy'] * 100)

  return optimizer, epoch_metrics
  ---
  def train_epoch(optimizer, train_ds, batch_size, epoch, rng):
    """Train for a single epoch."""
    train_ds_size = len(train_ds['image'])
    steps_per_epoch = train_ds_size // batch_size
    # 4 cycles per epoch #!
    learning_rate_fn = create_triangular_schedule( #!
      3e-3, 3e-2, steps_per_epoch // 4) #!
    perms = jax.random.permutation(rng, len(train_ds['image']))
    perms = perms[:steps_per_epoch * batch_size]
    perms = perms.reshape((steps_per_epoch, batch_size))
    batch_metrics = []
    for perm in perms:
      batch = {k: v[perm, ...] for k, v in train_ds.items()}
      optimizer, metrics = train_step(optimizer, batch, learning_rate_fn) #!
      batch_metrics.append(metrics)

    # compute mean of metrics across each batch in epoch.
    batch_metrics = jax.device_get(batch_metrics)
    epoch_metrics = {
        k: np.mean([metrics[k] for metrics in batch_metrics])
        for k in batch_metrics[0]}

    logging.info('train epoch: %d, loss: %.4f, accuracy: %.2f', epoch,
                epoch_metrics['loss'], epoch_metrics['accuracy'] * 100)

    return optimizer, epoch_metrics
