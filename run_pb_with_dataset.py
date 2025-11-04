"""Run a frozen TensorFlow graph (.pb) with data streamed from an in-memory dataset.

This example targets TensorFlow 1.x style execution with `tf.compat.v1.Session`.
It demonstrates how to:

1. Generate multiple feature tensors entirely in RAM.
2. Build a `tf.data.Dataset` that yields a batch of values for every input tensor.
3. Feed the batch into a graph restored from a `.pb` file.

Update the `INPUT_FEED_MAP` and `OUTPUT_TENSOR_NAMES` constants so they match the
placeholder and fetch tensor names in your frozen graph.
"""

from __future__ import annotations

import argparse
from typing import Dict, Iterable, Tuple

import numpy as np
import tensorflow as tf

tf.compat.v1.disable_eager_execution()


# Map short aliases to tensor names in the frozen graph.
# Replace the values with the exact names (including ":0") from your model.
INPUT_FEED_MAP: Dict[str, str] = {
    "feature_dense": "serving_default_dense_input:0",
    "feature_context": "serving_default_context_input:0",
}

# Replace these with the tensors you want to fetch when running the graph.
OUTPUT_TENSOR_NAMES: Tuple[str, ...] = (
    "StatefulPartitionedCall:0",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("graph_pb", type=str, help="Path to frozen graph (.pb) file")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=1)
    return parser.parse_args()


def generate_in_memory_samples(num_samples: int) -> Dict[str, np.ndarray]:
    """Create synthetic arrays for each model input in memory.

    Each array must share the same leading dimension (num_samples) so that it can
    be zipped into a dataset. Customize this function to mirror your model's
    expected shapes and dtypes.
    """

    rng = np.random.default_rng(seed=20241104)

    # Example features with distinct content for every input tensor.
    feature_dense = rng.normal(loc=0.0, scale=1.0, size=(num_samples, 16)).astype(np.float32)
    feature_context = rng.uniform(low=0.0, high=1.0, size=(num_samples, 4)).astype(np.float32)

    return {
        "feature_dense": feature_dense,
        "feature_context": feature_context,
    }


def build_dataset(features: Dict[str, np.ndarray], batch_size: int, num_epochs: int) -> Tuple[tf.compat.v1.data.Iterator, Dict[str, tf.Tensor]]:
    """Create an initializable iterator that yields batches for every feature."""

    dataset = tf.data.Dataset.from_tensor_slices(features)
    dataset = dataset.shuffle(buffer_size=len(next(iter(features.values()))))
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)

    iterator = tf.compat.v1.data.make_initializable_iterator(dataset)
    next_element = iterator.get_next()
    return iterator, next_element


def load_graph(pb_path: str) -> tf.Graph:
    graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(pb_path, "rb") as f:
        graph_def.ParseFromString(f.read())

    graph = tf.Graph()
    with graph.as_default():
        tf.import_graph_def(graph_def, name="")
    return graph


def get_tensors(graph: tf.Graph, tensor_names: Iterable[str]) -> Tuple[tf.Tensor, ...]:
    return tuple(graph.get_tensor_by_name(name) for name in tensor_names)


def main() -> None:
    args = parse_args()

    graph = load_graph(args.graph_pb)
    input_aliases = tuple(INPUT_FEED_MAP.keys())

    samples = generate_in_memory_samples(num_samples=args.num_samples)

    missing_aliases = set(input_aliases) - set(samples)
    if missing_aliases:
        raise ValueError(f"Missing feature arrays for aliases: {sorted(missing_aliases)}")

    with graph.as_default():
        iterator, next_batch_tensors = build_dataset(
            features=samples, batch_size=args.batch_size, num_epochs=args.num_epochs
        )

    feed_tensors = {alias: graph.get_tensor_by_name(name) for alias, name in INPUT_FEED_MAP.items()}
    fetch_tensors = get_tensors(graph, OUTPUT_TENSOR_NAMES)

    with tf.compat.v1.Session(graph=graph) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        sess.run(iterator.initializer)

        step = 0
        try:
            while True:
                batch_values = sess.run(next_batch_tensors)

                feed_dict = {
                    feed_tensors[alias]: batch_values[alias]
                    for alias in input_aliases
                }

                outputs = sess.run(fetch_tensors, feed_dict=feed_dict)

                print(f"step={step:04d}")
                for idx, output_value in enumerate(outputs):
                    print(f"  output[{idx}] shape={output_value.shape}")
                step += 1
        except tf.errors.OutOfRangeError:
            print("Dataset exhausted; finished running session.")


if __name__ == "__main__":
    main()
