Changelog
----------

vNext
------

(Add your change to a random empty line to avoid merge conflicts)
 -
 -
 -
 -
 -
 -
 - NamedTuples are no longer converted to tuples on assingment to a `linen.Module`.
 -
 -
 - Fixed a bug in FrozenDict which didn't allow copying dicts with reserved names.
 -
 -
 -
 - `flax.linen.enable_named_call` and `flax.linen.disable_named_call` now work anywhere instead of only affecting Modules constructed after the enable/disable call. Additionally, there is now `flax.linen.override_named_call` that provided a context manager to locally disable/enable named_call.
 -
 -
 - Fix the serialization of named tuples. Tuple fields are no longer stored in the state dict and the named tuple class is no longer recreated ([bug](https://github.com/google/flax/issues/1429)).
 -
 - linen Modules and dataclasses made with `flax.struct.dataclass` or `flax.struct.PyTreeNode` are now correctly recognized as dataclasses by static analysis tools like PyLance. Autocomplete of constructors has been verified to work with VSCode.
 -
 -
 - `flax.linen.Conv` no longer interprets an int past as kernel_size as a 1d convolution. Instead a type error is raised stating that
   a tuple/list should be provided. Stride and dilation arguments do support broadcasting a single int value now because this is not
   ambigious when the kernel rank is known. 
 - 

0.3.4
------

Possibly breaking changes:
 - When calling `init` the 'intermediates' collection is no longer mutable.
   Therefore, intermediates will no longer be returned from initialization by default. 
 - Don't update batch statistics during initialization.
 - When not using any non-determinism (e.g., dropout), it is not longer necessary to specify the `deterministic` argument in `MultiHeadDotProductAttention`.


Other changes:
 - Rewrote various examples to use Optax instead of Flax optimizers (e.g., Imagenet, SST2).
 - Added an NLP text classification example (on the SST-2 dataset) to
   [`examples/sst2`](https://github.com/google/flax/tree/main/examples/sst2).
   that uses a bidirectional LSTM (BiLSTM) to encode the input text.
 - Added `flax.training.train_state` to simplify using Optax optimizers.
 - `mutable` argument is now available on `Module.init` and `Module.init_with_outputs`
 - Bug fix: Correctly handle non-default parameters of Linen Modules with nested inheritance.
 - Expose `dot_product_attention_weights`, allowing access to attention weights.
 - `BatchNorm` instances will behave correctly during init when called multiple times.
 - Added a more extensive "how to contribute" guide in `contributing.md`.
 - Add proper cache behavior for [`lift.jit`](https://flax.readthedocs.io/en/latest/_autosummary/flax.linen.jit.html#flax.linen.jit),
fixing cache misses.
 - Fix bug in Embed layer: make sure it behaves correctly when embedding is np.array.
 - Fix `linen.Module` for deep inheritance chains.
 - Fix bug in DenseGeneral: correctly expand bias to account for batch & noncontracting dimensions.
 - Allow Flax lifted transforms to work on partially applied Modules.
 - Make `MultiOptimizer` use `apply_gradient` instead of `apply_param_gradient`.

0.3.3
------

Possible breaking changes:
 - Bug Fix: Disallow modifying attributes in Modules after they are initialized.
 - Raise an error when saving a checkpoint which has a smaller step than the
   latest checkpoint already saved.
 - MultiOptimizer now rejects the case where multiple sub optimizers update the
   same parameter.
  
Other changes:
 - Added custom error classes to many Linen errors. See: 
   https://flax.readthedocs.io/en/latest/flax.errors.html
 - Adds `Module.bind` for binding variables and RNGs to an interactive Module.
 - Adds `nn.apply` and `nn.init` for transforming arbitrary functions that take a `linen.Module` as their first argument.
 - Add option to overwrite existing checkpoints in `save_checkpoint`.
 - Remove JAX omnistaging check for forward compatibility.
 - Pathlib compatibility for checkpoint paths.
 - `is_leaf` argument in `traverse_util.flatten_dict`

0.3.2
------

`flax.nn` deprecation message no longer appears if you import flax directly.

NOTE: You must now explicitly import `flax.nn` if you want to use the old
      pre-Linen `flax.nn.Module`.

0.3.1
------

Many improvements to Linen, and the old `flax.nn` is officially reprecated! 

Notably, there's a clean API for extracting intermediates from modules
defined using `@nn.compact`, a more ergonomic API for using Batch Norm and Dropout in modules
defined using `setup`, support for `MultiOptimizer` with Linen, and multiple safety, performance
and error message improvements.

Possible breaking changes:
 - Call setup lazily. See #938 for motivation and more details.
 - Linen `Module` instances are now frozen after `setup` has been called.
   Previously mutations after setup could be dropped silently. Now the stateless requirement
   is enforced by raising a TypeError in `__setattr__` after `setup`.
 - Pytrees of dicts and lists are transformed into FrozenDict and tuples during
   attribute assignment.
   This avoids undetected submodules and inner state. 
 - Bug Fix `flax.core.apply` and `Module.apply`. Now it returns a tuple
   containing the output and a frozen empty
   collection when `mutable` is specified as an empty list.
 - `broadcast_dims` is now a attribute to `Dropout` instead of a `__call__`
   argument.
 - `use_running_average` and `deterministic` no longer have a default. They
   should be passed explicitly
 - Bug Fix `Scope.variable` mutability check, before a variable could only be
   initialized if the 'params' collection was mutable.

Other Improvements:
 - Re-introduced the `lm1b` language modeling example
 - Recognizes batch free inputs in pooling layers. (for use with vmap)
 - Add Adadelta optimizer
 - Fully deprecate all "pre-Linen" `flax.nn` classes and methods.
 - Some Module arguments can now be passed either as dataclass attribute or
   as argument to `__call__`. See [design note](https://flax.readthedocs.io/en/latest/design_notes/arguments.html)
 - Add `sow` method to `Module` and `capture_intermediates` argument to `Module.apply`.
   See [howto](https://flax.readthedocs.io/en/latest/howtos/extracting_intermediates.html) for usage patterns.
 - Support passing in modules directly as attributes to other modules, and
   deal with them correctly both in top-level modules and in submodules.
 - Don't require the `variable` argument to `Module.apply` to be a FrozenDict
 - Add support for dict/FrozenDict when using `ModelParamTraversal`
   As a result `MultiOptimizer` can be used properly with linen modules.
 - Added OptimizedLSTM: ~33% faster than the original LSTM when using <=1024 units
 - Fix dtype handling for Adam and LAMB optimizers in 64bit mode.
 - Added `is_mutable()` method to `Variable` and `is_mutable_collection()` to `flax.linen.Module`.
 - Add `axis_name` arg to `flax.linen.vmap`
 - Enable broadcast in `flax.linen.scan`
 - Fix behavior when inner module classes were defined in another module
 - Add automatic giant array chunking in msgpack checkpoints.
 - Log info message when a checkpoint is not found in the directory.

v0.3
-----
Linen is now out of Alpha (flax.nn is being deprecated)!

 - `flax.core.apply` and linen `Module.apply` will now only return the variables
   collections that were specified as mutable.
 - Fixed handling of multiple separate subclasses of a Module.
 - We now allow assignment of mixed Module pytrees in setup.
 - Refactored collection creation to fail early when modifying an undefined collection as
   before an non-existing non-mutable collection would just be silently ignored.
 - Added the silu activation function.
 - Add offset argument to Adafactor optimizer for fine-tuning schedules.
 - Relaxed limit on calling methods on unbound modules.
 - Relaxed parameter attribute check
 - Added centered version of RMSProp.
 - Added GCE getting started kit.
 - Renamed -gpu_type to -accelerator_type.
 - Fixed bug in MultiOptimizer causing it to throw away empty dictionary

### Improvements
 - Made FrozenDict constructor freeze correctly.
 - Made freeze a synonym of the FrozenDict constructor
 - Optimize freezing FrozenDicts by sharing immutable internal state.
 - We simplified __setattr__ handling of trees with Modules.
 - Minor improvements in dtype handling, broadcast option for dropout.
 - Added a dtype specification to Embed layer, made Adafactor use float32
   state consistently, and added a broadcasting option to the Dropout layer.
 - Improved frozen dict performance.
 - (Massive) docs improvements
 - End to end benchmarks added.
 - Examples were updated to Linen.

v0.2.2
----
 - Added Reinforcement Learning example (examples/ppo).
 - Fix Adafactor bug that prevented factorization.
 - Fix scan broadcast issue in functional core.
 - Fix initialization RNGs to work with omnistaging for jitted inits.
 - Replaces usage of 'param' kind to 'params' collection.
 - Fix LARS optimizer for zero param initialization.
 - Added various examples in Linen API. See [README.md](https://github.com/google/flax/blob/main/flax/linen/README.md) for more information.
 - Full JAX omnistaging compatibility.

v0.2
----
 - Added JAX trace-level checks for transforms.
 - BatchNorm added axis_index_groups for control in parallel training.
 - Optimizers broken out into separate directory with base class and implementations.
 - traverse_util added flatten_dict and unflatten_dict utility methods for nested dicts.

v0.1
----

### API Changes
 - Add ConvTranspose Module to nn.linear
 - Rename the following optional arguments to nn.linear.Conv:
     `lhs_dilation` -> `input_dilation`,
     `rhs_dilation` -> `kernel_dilation`
 - Change default layer names from numbers '0', '1', etc. to
   include the Module class name, e.g. 'Dense_0', 'LayerNorm_1'.
