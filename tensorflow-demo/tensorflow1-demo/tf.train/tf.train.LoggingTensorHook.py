# noinspection DuplicatedCode
import tensorflow as tf
from sklearn.datasets import load_iris

tf.reset_default_graph()

iris = load_iris()


def model_fn_(features, labels, mode, params):
    dense_layer = tf.layers.Dense(units=3, activation=None,
                                  kernel_initializer=tf.truncated_normal_initializer,
                                  name="dense")

    y = tf.identity(dense_layer(features), name='y')

    probas = tf.nn.softmax(y, name="probas")

    predictions = {"y": y, "proba": probas}
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    else:
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=y)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001)

        train_op = optimizer.minimize(loss, tf.train.get_or_create_global_step())

        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op,
            predictions=predictions)


def input_fn_():
    features = tf.data.Dataset.from_tensor_slices(iris.data)  # [4,]
    labels = tf.data.Dataset.from_tensor_slices(iris.target)  # []

    ds = tf.data.Dataset.zip((features, labels))

    ds = ds.batch(2)

    return ds


def main():
    config = tf.estimator.RunConfig().replace(log_step_count_steps=5, save_checkpoints_steps=5, keep_checkpoint_max=10)
    model = tf.estimator.Estimator(model_fn=model_fn_, config=config)

    log_tensors = {"dense_kernel": "dense/kernel", "y_": "y"}
    tf.logging.set_verbosity(tf.logging.INFO)
    hooks = [tf.train.LoggingTensorHook(tensors=log_tensors, every_n_iter=5)]
    model.train(input_fn=input_fn_, hooks=hooks, max_steps=100)


if __name__ == '__main__':
    main()
    print("done")
