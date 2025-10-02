import argparse
from flax import nnx
import jax
import numpy
from pathlib import Path
from orbax import checkpoint as ocp
from train import DigitClassifier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("one", type=Path, help="First checkpoint for comparison.")
    parser.add_argument("two", type=Path, help="Second checkpoint for comparison.")
    args = parser.parse_args()

    models = []
    for path in [args.one, args.two]:
        abstract_model = nnx.eval_shape(lambda: DigitClassifier(rngs=nnx.Rngs(0)))
        checkpointer = ocp.StandardCheckpointer()
        model = checkpointer.restore(path.resolve(), abstract_model)
        models.append(model)

    jax.tree.map(numpy.testing.assert_array_equal, *models)


if __name__ == "__main__":
    main()
