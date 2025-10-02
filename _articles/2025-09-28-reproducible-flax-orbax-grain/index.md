---
title: Reproducible Training with Flax, Orbax, and Grain
abstract: |
    Reproducible training of machine learning models is critical for preventing accidental performance regression, isolating the sensitivity of models to hyperparameter changes, and iterating on and improving models. However, guaranteeing a reproducible training pipeline is challenging due to the interaction of model initialization, stochastic mini-batches, and tracking optimizer state. This article demonstrates how to build a reproducible training script in the [JAX](https://github.com/jax-ml/jax) ecosystem using the modeling library [Flax](https://github.com/google/flax), checkpointing library [Orbax](https://github.com/google/orbax), and data loading library [Grain](https://github.com/google/grain).
author: Till Hoffmann
layout: article
---

Reproducibility of machine learning models is critical for preventing accidental performance regression, isolating the sensitivity of models to hyperparameter changes, and iterating on and improving models. Kapoor and Narayanan (2023) argue that there is a new reproducibility crisis upon us[^kapoor-reproducibility]. Building reproducible training pipelines in particular is challenging due to the interaction of model initialization, stochastic mini-batches, and tracking optimizer state. While the reproducibility crisis can by no means be addressed by reproducible training alone, it is a necessary step. This article demonstrates how to build a reproducible training script[^reproducibility-caveat] in the [JAX](https://github.com/jax-ml/jax) ecosystem using the modeling library [Flax](https://github.com/google/flax), checkpointing library [Orbax](https://github.com/google/orbax), and data loading library [Grain](https://github.com/google/grain). The latter two are used by Google in [the development](https://github.com/google-deepmind/gemma) of their flagship open-weights model [Gemma](https://deepmind.google/models/gemma/).

By "reproducible training pipeline," we mean that the pipeline can be run multiple times and yields the same model weights given the same pseudo-random number generator seed. The pipeline should also be interruptible and resumable. Orchestrating such a training pipeline is not difficult in principle (assuming we use a single machine), but its implementation can be fiddly. Here, we will develop such a pipeline, broadly following these steps:

1. Given a *single* seed to control randomness, we create Flax pseudo-random number generator streams [`flax.nnx.Rngs`](https://flax.readthedocs.io/en/latest/guides/randomness.html) for model initialization, data preprocessing and loading, and train-time randomness such as [drop-out](https://en.wikipedia.org/wiki/Dilution_(neural_networks)) or stochastic loss functions.
2. Create a data pipeline for the training and validation sets. Reproducible preparation and loading of the validation set is essential for early stopping and adaptive learning rate schedules.
3. Initialize the model and optimizer.
4. Run training for the desired number of epochs and save checkpoints along the way as well as the final model.
5. Resume training from a checkpoint when the script starts.

At the end of this article, we will end up with a command line interface (CLI) with the following arguments.

```
usage: train [-h] [--seed SEED] [--checkpoint-every CHECKPOINT_EVERY]
             output num_epochs

positional arguments:
  output                directory for saving checkpoints or to resume from
  num_epochs            number of training epochs

options:
  -h, --help            show this help message and exit
  --seed SEED           pseudo-random number generator seed
  --checkpoint-every CHECKPOINT_EVERY
                        checkpoint every N steps
```

## Basic Training Script

### Handling Random State

Given a *single* seed from the CLI, we first initialize a `main_key`, a [random number generator state used by JAX](https://docs.jax.dev/en/latest/random-numbers.html). We then derive all random state from the `main_key` by splitting it into a `model_key` for parameter initialization and drop-out and a `train_key` responsible for all training-related randomness, such as minibatch shuffling, data augmentation (like random image crops, rotations, or flips), and stochastic loss functions (such as sampled softmax loss or drawing samples from normalizing flows to train density estimators).

```python
>>> from flax import nnx
>>> from jax import random


>>> args = ...  # Parse from command line arguments.
>>> main_key = random.key(args.seed)
>>> model_key, train_key = random.split(main_key)
>>> train_rngs = nnx.Rngs(train_key)
```

### Creating a Data Pipeline With Grain

[Grain](https://google-grain.readthedocs.io/), the data preprocessing and loading library we will use, [offers two interfaces](https://google-grain.readthedocs.io/en/latest/api_choice.html): [`DataLoader`](https://google-grain.readthedocs.io/en/stable/tutorials/data_loader_tutorial.html) and [`Dataset`](https://google-grain.readthedocs.io/en/stable/tutorials/dataset_basic_tutorial.html). The former has an interface that is immediately familiar to PyTorch users but offers less flexibility. The latter "is a low-level API that uses chaining syntax to define data transformation steps." We use it here to load the classic MNIST dataset for digit classification using [`tensorflow_datasets`](https://www.tensorflow.org/datasets) as the source data.

```python
>>> import tensorflow_datasets as tfds


>>> train_data_source = tfds.load("mnist", split="train")
>>> train_dataset = (
...     MapDataset.source(train_data_source)
...     .seed(seed_from_key(train_rngs()))
...     .shuffle()
...     .repeat(args.num_epochs)
...     .batch(64, drop_remainder=True)
... )
```

The [`MapDataset`](https://google-grain.readthedocs.io/en/stable/grain.dataset.html#grain.MapDataset) wraps the underlying MNIST data source and supports random-access to its elements. Calling [`seed`](https://google-grain.readthedocs.io/en/stable/grain.dataset.html#grain.MapDataset.seed) in the chain instantiates the random state used to then [`shuffle`](https://google-grain.readthedocs.io/en/stable/grain.dataset.html#grain.MapDataset.shuffle) the dataset and further optional [`random_map`](https://google-grain.readthedocs.io/en/stable/grain.dataset.html#grain.MapDataset.random_map) transformations we may want to apply, e.g., for data augmentation. We [`repeat`](https://google-grain.readthedocs.io/en/stable/grain.dataset.html#grain.MapDataset.repeat) the dataset for the desired number of epochs before assembling [`batch`](https://google-grain.readthedocs.io/en/stable/grain.dataset.html#grain.MapDataset.batch)es. We use `drop_remainder=True` to drop the final batch if it does not have the desired number of elements. This avoids possibly high-variance gradients for very small batches.

We used the function `seed_from_key` to get a numerical seed from a JAX random key because that's what Grain expects (it uses [`numpy.random.Generator`](https://numpy.org/doc/2.1/reference/random/generator.html) internally). We could have initialized a separate integer-based random state or passed a `--data-seed` CLI argument, but encapsulating all random state in `train_rngs` simplifies things. `seed_from_key` simply computes the XOR of the two `uint32` parts that constitute a JAX random key. It is defined as follows.

```python
>>> def seed_from_key(key):
...     a, b = random.key_data(key)
...     return int(a ^ b)
```

Here is one batch from the training set.

```python
>>> train_data_iterator = iter(train_dataset)
>>> batch = next(train_dataset)
>>> batch
{
    'image': array([....], shape=(64, 28, 28, 1), dtype=uint8),
    'label': array([3, 0, ..., 4, 3], shape=(64,), dtype=uint64),
}
```

As desired, the batch size is 64, we have 28 by 28 pixel monochrome input images, and integer labels. The `train_data_iterator` is a stateful iterator over the dataset which we will later checkpoint for reproducibility. It's time to build a model.

### Initializing the Model and Optimizer

We use a simple convolutional neural network whose outputs are logits for each of the ten classes representing digits. The architecture is straight from the [Flax tutorial](https://flax.readthedocs.io/en/v0.8.3/experimental/nnx/mnist_tutorial.html#define-the-network-with-nnx).

```python
>>> class DigitClassifier(nnx.Module):
...     def __init__(self, *, rngs: nnx.Rngs):
...         self.conv1 = nnx.Conv(1, 32, kernel_size=(3, 3), rngs=rngs)
...         self.conv2 = nnx.Conv(32, 64, kernel_size=(3, 3), rngs=rngs)
...         self.linear1 = nnx.Linear(3136, 256, rngs=rngs)
...         self.linear2 = nnx.Linear(256, 10, rngs=rngs)
...
...     def __call__(self, x):
...         # Rescale inputs from the original image domain [0, 255] to [0, 1].
...         x = x / 255
...         # Apply two convolutional layers with average pooling and downsampling.
...         x = nnx.avg_pool(nnx.relu(self.conv1(x)), window_shape=(2, 2), strides=(2, 2))
...         x = nnx.avg_pool(nnx.relu(self.conv2(x)), window_shape=(2, 2), strides=(2, 2))
...         # Flatten features and apply two dense layers.
...         x = x.reshape(x.shape[0], -1)
...         x = nnx.relu(self.linear1(x))
...         x = self.linear2(x)
...         return x
```

We use an Optax Adam optimizer with learning rate 0.001 and default parameters. The model and optimizer creation is wrapped in a function `create_model_and_optimizer`, and we'll come back to the need for a dedicated function when we load checkpoints.

```python
>>> import optax


>>> def create_model_and_optimizer(key):
...     rngs = nnx.Rngs(key)
...     model = DigitClassifier(rngs=rngs)
...     optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
...     return model, optimizer


>>> model, optimizer = create_model_and_optimizer(model_key)
```

### Training

Finally, what we have been building towards: Running the training loop. We use a `train_step` function to update model parameters. It evaluates the gradients of `loss_fn` and updates the model weights using the `optimizer`. `loss_fn` in turn applies the model to input images to obtain logits and evaluates the softmax cross-entropy loss to be minimized. Isolating the behavior in a function allows us to [JIT-compile](https://docs.jax.dev/en/latest/jit-compilation.html) the function with significant performance gains. One epoch takes about 12 seconds on a MacBook Air with JIT-compilation and 27 seconds without.

```python
>>> def train_step(model, optimizer, inputs, labels):
...    grad_fn = nnx.value_and_grad(loss_fn)
...    loss, grads = grad_fn(model, inputs, labels)
...    optimizer.update(model, grads)
...    return loss


>>> def loss_fn(model, inputs, labels):
...     logits = model(inputs)
...     pointwise_losses = optax.softmax_cross_entropy_with_integer_labels(
...         logits, labels
...     )
...     return pointwise_losses.mean()
```

Calling `train_step` updates the model weights *in place*, i.e., the attributes of the `model` instance are updated[^pure-note].

```python
>>> model.linear2.bias
Param( # 10 (40 B)
  value=Array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], dtype=float32)
)
>>> jitted_train_step(model, optimizer, batch["image"], batch["label"])
Array(2.3046489, dtype=float32)
>>> model.linear2.bias
Param( # 10 (40 B)
  value=Array([-0.000999, 0.000999, ..., -0.000999, -0.000999], dtype=float32)
)
```

Let's put it all together into a training loop that iterates over batches and updates the weights for each batch.

```python
>>> with tqdm(total=len(train_dataset)) as progress:
...    for batch in train_data_iterator:
...        loss = jitted_train_step(
...            state.model, state.optimizer, batch["image"], batch["label"]
...        )
...        progress.update()
...        progress.set_description(f"loss: {loss:.3f}")
```

We explicitly iterate over the `train_data_iterator` rather than implicitly over the `train_dataset` because we need to keep track of the `train_data_iterator` state. Running the script from the command line shows that the model is learning *something*, reducing the loss from 2.309 to 0.007 over three epochs.

```bash
$ uv run python train.py demo/one 3
loss: 0.007: 100%|██████████| 2812/2812 [00:36<00:00, 98.36it/s]
```

## Saving Checkpoints

Training scripts carry a lot of entangled state that is spread across data iterators, random number key streams (e.g., for drop-out), model weights, and optimizer state. They all need to be checkpointed consistently, and we assemble them into a `TrainState` object similar to [Flax Linen's approach](https://flax.readthedocs.io/en/latest/api_reference/flax.training.html#flax.training.train_state.TrainState).

```python
>>> from dataclasses import dataclass


>>> @dataclass
>>> class TrainState:
...     model: DigitClassifier
...     optimizer: nnx.Optimizer
...     rngs: nnx.Rngs
...     step: int
...     train_iterator: DatasetIterator
```

We will add periodic checkpointing, but we first need to construct the state.

```python
>>> if output.is_dir():
...     raise NotImplementedError("We'll implement checkpoint loading later.")
... else:
...     model, optimizer = create_model_and_optimizer(model_key)
...     state = TrainState(model, optimizer, train_rngs, 0, train_data_iterator)
```

The loop is then wrapped in a [`orbax.checkpoint.CheckpointManager`](https://orbax.readthedocs.io/en/latest/api_reference/checkpoint.checkpoint_manager.html) context manager. Orbax uses [asynchronous checkpointing](https://orbax.readthedocs.io/en/latest/guides/checkpoint/async_checkpointing.html) by default, and the context manager ensures checkpoints are saved before terminating the process. We pass `args.output.resolve()` to the manager because Orbax expects absolute paths.

```python
>>> from orbax import checkpoint as ocp


>>> with (
...     tqdm(total=len(train_dataset), initial=state.step) as progress,
...     ocp.CheckpointManager(args.output.resolve()) as checkpoint_manager,
... ):
...     for state.step, batch in enumerate(train_data_iterator, start=state.step):
...         loss = jitted_train_step(
...             state.model, state.optimizer, batch["image"], batch["label"]
...         )
...
...         if (
...             args.checkpoint_every
...             and state.step
...             and state.step % args.checkpoint_every == 0
...         ):
...             state.save(checkpoint_manager)
...             print(
...                 f"Saved checkpoint to '{output}' at step {state.step} with training loss {loss:.3f}."
...             )
...
...         progress.update()
...         progress.set_description(f"loss: {loss:.3f}")
...
...     state.save(checkpoint_manager)
...     print(
...         f"Saved final checkpoint to '{output}' at step {state.step} with training loss {loss:.3f}."
...     )
```

Note that we use `state.step` directly as the loop variable in `enumerate`, which automatically updates it with each iteration. Combined with `start=state.step`, this ensures the step counter is always correct whether starting fresh (state.step = 0) or resuming from a checkpoint. We also pass `initial=state.step` to `tqdm` so the progress bar reflects the correct position when resuming.

We save checkpoints every `args.checkpoint_every` steps and at the end of the loop. The `save` function remains to be implemented.

```python
>>> from grain.checkpoint import CheckpointSave as GrainCheckpointSave


>>> class TrainState:
...     ...
...
...     def save(self, manager, **kwargs):
...         args = {
...             "model": ocp.args.StandardSave(self.model),
...             "optimizer": ocp.args.StandardSave(self.optimizer),
...             "rngs": ocp.args.StandardSave(self.rngs),
...             "train_iterator": GrainCheckpointSave(self.train_iterator),
...         }
...         manager.save(self.step, args=ocp.args.Composite(**args), **kwargs)
```

The function constructs `ocp.args.Composite` args and uses the manager to store the data with the current step number. In Orbax, `args` come in two flavors: save `args` for storing information on disk, and restore `args` to load information from disk. Each `manager.save` call will store one of these `args` in a directory. `ocp.args.Composite` creates a directory for each of its members so we can later load the model separately from the optimizer or other training-specific information. The `StandardSave` arg is a flexible serializer that can handle a range of objects, including models based on `nnx.Module`, `nnx.Optimizer`, and `nnx.Rngs`. `GrainCheckpointSave` saves the state of the training data iterator, ensuring all components of the training state is persisted.

Let's run the script again.

```bash
$ rm -rf demo/one

$ uv run python train.py demo/one 3
loss: 0.007: 100%|██████████| 2812/2812 [00:36<00:00, 98.36it/s]
Saved final checkpoint to '.../demo/one' at step 2811 with training loss 0.007.

$ ls -l demo/one/2811
total 8
-rw-r--r--@ 1 till  staff  547 Oct  2 15:06 _CHECKPOINT_METADATA
drwxr-xr-x@ 8 till  staff  256 Oct  2 15:06 model
drwxr-xr-x@ 8 till  staff  256 Oct  2 15:06 optimizer
drwxr-xr-x@ 8 till  staff  256 Oct  2 15:06 rngs
drwxr-xr-x@ 3 till  staff   96 Oct  2 15:06 train_iterator
```

The `demo/one/2811` directory contains metadata, and one folder for each of the constituents of the `Composite`. We've saved our first checkpoint. We can load the model for deployment or continue training at a later stage, e.g., for a larger number of epochs.

## Resuming From Checkpoints

Checkpoints are not very useful if we cannot resume training—which we'll tackle now. We implement a `restore` method on the `TrainState` that mirrors the `save` method.

```python
>>> from grain.checkpoint import CheckpointRestore as GrainCheckpointRestore


>>> class TrainState:
...     ...
...
...     def restore(self, manager: ocp.CheckpointManager, **kwargs):
...          args = {
...              "model": ocp.args.StandardRestore(self.model),
...              "optimizer": ocp.args.StandardRestore(self.optimizer),
...              "rngs": ocp.args.StandardRestore(self.rngs),
...              "train_iterator": GrainCheckpointRestore(self.train_iterator),
...          }
...          restored = manager.restore(
...              step=self.step, args=ocp.args.Composite(**args), **kwargs
...          )
...          self.__dict__.update(step=self.step + 1, **restored)
```

We increment `self.step` by 1 after restoring because checkpoint N contains the model state after completing training step N. When we resume, we need to start training at step N+1 to avoid re-training the same step. This increment, combined with passing `start=state.step` to `enumerate` in the training loop, ensures we pick up exactly where we left off.

The `args` are identical to the `save` implementation except `...Save` is replaced by `...Restore`. `restored` is a dictionary-like object holding the restored checkpoint, and we use it to update the state in-place. However, `restore` is an *instance* method, so we already need to have a state to restore the state—a chicken and egg problem. Orbax requires a skeleton of the objects to restore so it knows how to populate the instance. Luckily `nnx.eval_shape` allows us to construct abstract models, optimizers, and `Rngs` without allocating the tensors.

```python
>>> if output.is_dir():
...     model, optimizer = nnx.eval_shape(lambda: create_model_and_optimizer(model_key))
...     with ocp.CheckpointManager(output) as checkpoint_manager:
...         step = checkpoint_manager.latest_step()
...         state = TrainState(model, optimizer, train_rngs, step, train_data_iterator)
...         state.restore(checkpoint_manager)
...         print(f"Restored checkpoint from '{output}' at step {state.step}.")
... else:
...     ...
```

Here, we create an abstract model and optimizer to populate the train state, and we pass in the `train_rngs` and `train_data_iterator`—their state will simply be overwritten by `restore`. Voilà, we have a training pipeline with checkpointing that can be resumed. But does it work?

## Comparing the Runs

To assess if the pipeline is resumable and reproducible, we repeat the previous training run but split it into two parts. We first train for two epochs and then, in a separate process, for a third.

```bash
$ uv run python train.py demo/two 2
loss: 0.017: 100%|██████████| 1875/1875 [00:24<00:00, 94.34it/s]
Saved final checkpoint to '.../demo/two' at step 1874 with training loss 0.017.

$ uv run python train.py demo/two 3
Restored checkpoint from '.../demo/two' at step 1875.
loss: 0.007: 100%|██████████| 2812/2812 [00:13<00:00, 94.10it/s]
Saved final checkpoint to '.../demo/two' at step 2811 with training loss 0.007.

$ ls -l demo/two
total 0
drwxr-xr-x@ 7 till  staff  224 Oct  2 15:06 1874
drwxr-xr-x@ 7 till  staff  224 Oct  2 15:06 2811
```

In our second experiment, the output directory contains two checkpoints: one stored after the second and one after the third epoch. If our pipeline works as intended, the weights of the two runs must match *exactly*. Let's load the model weights and compare them.

```python
>>> paths = ["demo/one/2811/model", "demo/two/2811/model"]
>>> models = []
>>> for path in paths:
...     abstract_model = nnx.eval_shape(lambda: DigitClassifier(rngs=nnx.Rngs(0)))
...     checkpointer = ocp.StandardCheckpointer()
...     model = checkpointer.restore(path.resolve(), abstract_model)
...     models.append(model)

>>> models[0].linear2.bias
Param( # 10 (40 B)
  value=Array([-0.01458673, -0.00075466, -0.00246617,  0.00097542, -0.00451297,
          0.01191687, -0.02906019, -0.00665942,  0.03023529, -0.00634232],      dtype=float32)
)
>>> models[1].linear2.bias
Param( # 10 (40 B)
  value=Array([-0.01458673, -0.00075466, -0.00246617,  0.00097542, -0.00451297,
          0.01191687, -0.02906019, -0.00665942,  0.03023529, -0.00634232],      dtype=float32)
)
```

The biases of the final layer are identical! We'll use `jax.tree.map` to compare all weights as a final check.

```python
>>> import jax
>>> import numpy


>>> jax.tree.map(numpy.testing.assert_array_equal, *models)
```

The code maps the assertion that the arrays are equal over all weights. It raises no errors, confirming that our pipeline works!

## Concluding Thoughts

We have implemented a reproducible training pipeline for training models with Flax, Grain, and Orbax! While a production-grade pipeline should include error checking, logging, handling checkpoint rotation, and more, this simple script of fewer than 200 lines incorporates all the important components for reproducibility and is ready for more complex training, e.g., injecting randomness into the `train_step` using the `train_rngs`.

[^reproducibility-caveat]: Reproducibility can sometimes not been guaranteed if computation is subject to randomness beyond the pseudo-random number generator streams we control. For example, computation on accelerators can [suffer from non-determinism](https://stackoverflow.com/a/50746919/1150961).
[^kapoor-reproducibility]: S Kapoor and A Narayanan. (2023) "Leakage and the reproducibility crisis in machine-learning-based science." *Patterns*, 4, 100804. doi: [10.1016/j.patter.2023.100804](https://doi.org/10.1016/j.patter.2023.100804)
[^pure-note]: This is at odds with JAX's philosophy that all operations should be pure and not have side effects. But this behavior is intuitive and mirrors the familiar behavior of PyTorch optimization.
