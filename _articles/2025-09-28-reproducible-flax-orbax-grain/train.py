import argparse
from dataclasses import dataclass
from flax import nnx
from grain import DatasetIterator, MapDataset
from grain.checkpoint import (
    CheckpointSave as GrainCheckpointSave,
    CheckpointRestore as GrainCheckpointRestore,
)
from jax import random
import optax
import orbax.checkpoint as ocp
from pathlib import Path
import tensorflow_datasets as tfds
from tqdm import tqdm


def seed_from_key(key):
    a, b = random.key_data(key)
    return int(a ^ b)


class DigitClassifier(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(1, 32, kernel_size=(3, 3), rngs=rngs)
        self.conv2 = nnx.Conv(32, 64, kernel_size=(3, 3), rngs=rngs)
        self.linear1 = nnx.Linear(3136, 256, rngs=rngs)
        self.linear2 = nnx.Linear(256, 10, rngs=rngs)

    def __call__(self, x):
        # Rescale inputs from the original image domain [0, 255] to [0, 1].
        x = x / 255
        # Apply two convolutional layers with average pooling and downsampling.
        x = nnx.avg_pool(nnx.relu(self.conv1(x)), window_shape=(2, 2), strides=(2, 2))
        x = nnx.avg_pool(nnx.relu(self.conv2(x)), window_shape=(2, 2), strides=(2, 2))
        # Flatten features and apply two dense layers.
        x = x.reshape(x.shape[0], -1)
        x = nnx.relu(self.linear1(x))
        x = self.linear2(x)
        return x


def create_model_and_optimizer(key):
    rngs = nnx.Rngs(key)
    model = DigitClassifier(rngs=rngs)
    optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
    return model, optimizer


def loss_fn(model, inputs, labels):
    logits = model(inputs)
    pointwise_losses = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
    return pointwise_losses.mean()


def train_step(model, optimizer, inputs, labels):
    grad_fn = nnx.value_and_grad(loss_fn)
    loss, grads = grad_fn(model, inputs, labels)
    optimizer.update(model, grads)
    return loss


@dataclass
class TrainState:
    model: DigitClassifier
    optimizer: nnx.Optimizer
    rngs: nnx.Rngs
    step: int
    train_iterator: DatasetIterator

    def save(self, manager: ocp.CheckpointManager, **kwargs):
        args = {
            "model": ocp.args.StandardSave(self.model),
            "optimizer": ocp.args.StandardSave(self.optimizer),
            "rngs": ocp.args.StandardSave(self.rngs),
            "train_iterator": GrainCheckpointSave(self.train_iterator),
        }
        manager.save(self.step, args=ocp.args.Composite(**args), **kwargs)

    def restore(self, manager: ocp.CheckpointManager, **kwargs):
        args = {
            "model": ocp.args.StandardRestore(self.model),
            "optimizer": ocp.args.StandardRestore(self.optimizer),
            "rngs": ocp.args.StandardRestore(self.rngs),
            "train_iterator": GrainCheckpointRestore(self.train_iterator),
        }
        restored = manager.restore(
            step=self.step, args=ocp.args.Composite(**args), **kwargs
        )
        # We have to advance the step because the checkpoint was saved *after*
        # optimization, and we do not want to run the step again.
        self.__dict__.update(step=self.step + 1, **restored)


def main() -> None:
    parser = argparse.ArgumentParser("train")
    parser.add_argument(
        "--seed", type=int, help="pseudo-random number generator seed", default=42
    )
    parser.add_argument("--checkpoint-every", type=int, help="checkpoint every N steps")
    parser.add_argument(
        "output", type=Path, help="directory for saving checkpoints or to resume from"
    )
    parser.add_argument("num_epochs", type=int, help="number of training epochs")
    args = parser.parse_args()

    # Step 1: Set up random number generator state. We use a `main_key` and derive all
    # randomness from it. The `model_key` controls model-related randomness like
    # parameter initialization and drop-out. The `train_rngs` are responsible for all
    # other randomness during the training process.
    main_key = random.key(args.seed)
    model_key, train_key = random.split(main_key)
    train_rngs = nnx.Rngs(train_key)

    # Step 2: Create data loader and iterator.
    train_data_source = tfds.data_source("mnist", split="train")
    train_dataset = (
        MapDataset.source(train_data_source)
        .seed(seed_from_key(train_rngs()))
        .shuffle()
        .repeat(args.num_epochs)
        .batch(64, drop_remainder=True)
    )
    train_data_iterator = iter(train_dataset)

    # Step 3: Initialize a model and optimizer if we are starting anew or restore from a
    # checkpoint if the directory exists.
    checkpoint_manager: ocp.CheckpointManager
    output: Path = args.output.resolve()
    if output.is_dir():
        model, optimizer = nnx.eval_shape(lambda: create_model_and_optimizer(model_key))
        with ocp.CheckpointManager(output) as checkpoint_manager:
            step = checkpoint_manager.latest_step()
            assert step is not None, (
                f"Output directory '{output}' exists but does not have checkpoints."
            )
            state = TrainState(model, optimizer, train_rngs, step, train_data_iterator)
            state.restore(checkpoint_manager)
            print(f"Restored checkpoint from '{output}' at step {state.step}.")
    else:
        model, optimizer = create_model_and_optimizer(model_key)
        state = TrainState(model, optimizer, train_rngs, 0, train_data_iterator)

    # Step 4: Run the training loop.
    jitted_train_step = nnx.jit(train_step)
    with (
        tqdm(total=len(train_dataset), initial=state.step) as progress,
        ocp.CheckpointManager(output) as checkpoint_manager,
    ):
        loss = float("nan")
        for state.step, batch in enumerate(train_data_iterator, start=state.step):
            loss = jitted_train_step(
                state.model, state.optimizer, batch["image"], batch["label"]
            )

            if (
                args.checkpoint_every
                and state.step
                and state.step % args.checkpoint_every == 0
            ):
                state.save(checkpoint_manager)
                print(
                    f"Saved checkpoint to '{output}' at step {state.step} with training loss {loss:.3f}."
                )

            progress.update()
            progress.set_description(f"loss: {loss:.3f}")

        # Save the final state.
        state.save(checkpoint_manager)
        print(
            f"Saved final checkpoint to '{output}' at step {state.step} with training loss {loss:.3f}."
        )


if __name__ == "__main__":
    main()
