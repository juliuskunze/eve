Examples
=============

Core examples
-------------

Each example is designed to be self-contained and easily forkable, while
reproducing relevant results in different areas of machine learning. These
examples adhere to a shared style and functionality outlined in `#231`_. All
examples are under the folder `flax/examples/
<https://github.com/google/flax/tree/main/examples/>`__. Some of the
examples below have a link [Interactive] that lets you run them directly in
Colab.

.. _#231: https://github.com/google/flax/issues/231


Image classification

   -  `MNIST <https://github.com/google/flax/tree/main/examples/mnist/>`__ [`Interactive
      <https://colab.research.google.com/github/google/flax/blob/main/examples/mnist/mnist.ipynb>`__] :
      Convolutional neural network for MNIST classification (featuring simple code).
   -  `ImageNet <https://github.com/google/flax/tree/main/examples/imagenet/>`__ :
      Resnet-50 on imagenet with weight decay (featuring multi host SPMD, custom
      preprocessing, checkpointing, dynamic scaling, mixed precision).

Reinforcement Learning

   -  `Proximal Policy
      Optimization <https://github.com/google/flax/tree/main/examples/ppo/>`__ :
      Learning to play Atari games (featuring single host SPMD, RL setup).

Natural language processing

   -  `Sequence to sequence for number
      addition <https://github.com/google/flax/tree/main/examples/seq2seq/>`__
      (featuring simple code, LSTM state handling, on the fly data generation).
   -  `Parts-of-speech tagging
      <https://github.com/google/flax/tree/main/examples/nlp_seq/>`__ : Simple
      transformer encoder model using the universal dependency dataset.
   -  `Sentiment classification
      <https://github.com/google/flax/tree/main/examples/sst2/>`__ : 
      with a LSTM model.
   -  `Transformer encoder/decoder model trained on WMT
      <https://github.com/google/flax/tree/main/examples/wmt/>`__ :
      Translating English/German (featuring multihost SPMD, dynamic bucketing,
      attention cache, packed sequences, recipe for TPU training on GCP).
   -  `Transformer encoder trained on one billion word benchmark
      <https://github.com/google/flax/tree/main/examples/lm1b/>`__ :
      for autoregressive language modeling, based on the WMT example above.

Generative models

   -  `Variational
      auto-encoder <https://github.com/google/flax/tree/main/examples/vae/>`__ :
      Trained on binarized MNIST (featuring simple code, vmap).
   -  `PixelCNN++ <https://github.com/google/flax/tree/main/examples/pixelcnn/>`__ :
      Trained on cifar10 (featuring single host SPMD, checkpointing, Polyak decay).


Community Examples
--------------------------------

In addition to the curated list of official Flax examples, there is a growing
community of people using Flax to build new types of machine learning models. We
are happy to showcase any example built by the community here! If you want to
submit your own example, we suggest that you start by forking one of the
official Flax example, and start from there.

Using Linen
~~~~~~~~~~~~~~~~~~~~

+-------------------------------+---------------------+------------------------+--------------------------------------+
|             Link              |       Author        |       Task type        |              Reference               |
+===============================+=====================+========================+======================================+
| `matthias-wright/flaxmodels`_ | `@matthias-wright`_ | Various                | Various                              |
+-------------------------------+---------------------+------------------------+--------------------------------------+
| `google/vision_transformer`_  | `@andsteing`_       | Image classification,  | https://arxiv.org/abs/2010.11929 and |
|                               |                     | fine-tuning            | https://arxiv.org/abs/2105.01601     |
+-------------------------------+---------------------+------------------------+--------------------------------------+
| `JAX-RL`_                     | `@henry-prior`_     | Reinforcement learning | N/A                                  |
+-------------------------------+---------------------+------------------------+--------------------------------------+
| `DCGAN`_ Colab                | `@bkkaggle`_        | Image Synthesis        | https://arxiv.org/abs/1511.06434     |
+-------------------------------+---------------------+------------------------+--------------------------------------+
| `BigBird Fine-tuning`_        | `@vasudevgupta7`_   | Question-Answering     | https://arxiv.org/abs/2007.14062     |
+-------------------------------+---------------------+------------------------+--------------------------------------+

.. _`matthias-wright/flaxmodels`: https://github.com/matthias-wright/flaxmodels
.. _`google/vision_transformer`: https://github.com/google-research/vision_transformer
.. _`JAX-RL`: https://github.com/henry-prior/jax-rl
.. _`DCGAN`: https://github.com/bkkaggle/jax-dcgan
.. _`BigBird Fine-tuning`: https://github.com/huggingface/transformers/tree/master/examples/research_projects/jax-projects/big_bird
.. _`@matthias-wright`: https://github.com/matthias-wright
.. _`@andsteing`: https://github.com/andsteing
.. _`@bkkaggle`: https://github.com/bkkaggle
.. _`@vasudevgupta7`: https://github.com/vasudevgupta7

Using the Deprecated ``flax.nn`` API
~~~~~~~~~~~~~~~~~~~~

The following examples were created using the old pre-Linen API. You can still
look at them for inspiration. Or maybe ask the authors if they would accept a
pull request to the new API if you want to earn some Flax karma?

+----------------------------------+-----------------+------------------------+----------------------------------+
|               Link               |     Author      |       Task type        |            Reference             |
+==================================+=================+========================+==================================+
| `Gaussian Processes regression`_ | `@danieljtait`_ | Regression             | N/A                              |
+----------------------------------+-----------------+------------------------+----------------------------------+
| `DQN`_                           | `@joaogui1`_    | Reinforcement learning | N/A                              |
+----------------------------------+-----------------+------------------------+----------------------------------+
| `Various CIFAR SOTA Models`_     | `@PForet`_      | Image Classification   | N/A                              |
+----------------------------------+-----------------+------------------------+----------------------------------+

.. _`Gaussian Processes regression`: https://github.com/danieljtait/ladax/tree/master/examples
.. _`DQN`: https://github.com/joaogui1/RL-JAX/tree/master/DQN
.. _`Various CIFAR SOTA Models`: https://github.com/google-research/google-research/tree/master/flax_models/cifar
.. _`DCGAN`: https://github.com/bkkaggle/jax-dcgan
.. _`@danieljtait`: https://github.com/danieljtait
.. _`@henry-prior`: https://github.com/henry-prior
.. _`@joaogui1`: https://github.com/joaogui1
.. _`@PForet`: https://github.com/PForet

More examples
-------------

**Looking for "FOO" implemented in Flax?** We use GitHub issues to keep track of
which models people are most interested in seeing re-implemented in Flax. If you
can't find what you're looking for in the `list "example requested"`_, file an
issue with this template_. If the model you are looking for has already been
requested by others, upvote the issue to help us see which ones are the most
requested.

**Looking to implement something in Flax?** Consider looking at the `list
"example requested"`_ and go ahead and build it!

.. _`list "example requested"`: https://github.com/google/flax/labels/example%20request
.. _template: https://github.com/google/flax/issues/new?assignees=&template=example_request.md&title=
