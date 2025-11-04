"""Example TensorFlow 1.x style training script using an in-memory dataset.

This script shows how to generate synthetic data directly in memory, wrap it
into a `tf.data.Dataset`, and run a simple training loop with
`tf.compat.v1.Session`.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf


def build_dataset(batch_size: int, num_epochs: int) -> tf.data.Dataset:
    """Create a dataset backed by in-memory numpy arrays."""

    # Generate toy regression data y = 3 * x + noise.
    rng = np.random.default_rng(seed=42)
    x_data = np.linspace(-1.0, 1.0, num=200, dtype=np.float32)
    noise = rng.normal(loc=0.0, scale=0.1, size=x_data.shape).astype(np.float32)
    y_data = 3.0 * x_data + noise

    dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data))
    dataset = dataset.shuffle(buffer_size=len(x_data))
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    return dataset


def build_model(features: tf.Tensor) -> tf.Tensor:
    """Define a simple linear regression model."""

    w = tf.compat.v1.get_variable("weight", shape=(), initializer=tf.zeros_initializer())
    b = tf.compat.v1.get_variable("bias", shape=(), initializer=tf.zeros_initializer())
    predictions = w * features + b
    return predictions


def main() -> None:
    tf.compat.v1.disable_eager_execution()

    batch_size = 32
    num_epochs = 100
    learning_rate = 0.1

    dataset = build_dataset(batch_size=batch_size, num_epochs=num_epochs)
    iterator = tf.compat.v1.data.make_initializable_iterator(dataset)
    next_features, next_labels = iterator.get_next()

    with tf.compat.v1.variable_scope("linear_regression"):
        preds = build_model(next_features)

    loss = tf.reduce_mean(input_tensor=tf.square(preds - next_labels))
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(loss)

    init_op = tf.compat.v1.global_variables_initializer()

    with tf.compat.v1.Session() as sess:
        sess.run(init_op)
        sess.run(iterator.initializer)

        step = 0
        try:
            while True:
                _, loss_value = sess.run([train_op, loss])
                if step % 50 == 0:
                    print(f"step={step:04d}, loss={loss_value:.6f}")
                step += 1
        except tf.errors.OutOfRangeError:
            pass

        learned_w, learned_b = sess.run([
            tf.compat.v1.get_variable("linear_regression/weight", shape=()),
            tf.compat.v1.get_variable("linear_regression/bias", shape=()),
        ])

        print("Training finished.")
        print(f"Learned weight: {learned_w:.3f}")
        print(f"Learned bias: {learned_b:.3f}")


if __name__ == "__main__":
    main()
