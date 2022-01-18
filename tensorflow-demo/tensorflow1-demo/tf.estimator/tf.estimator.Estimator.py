import tensorflow as tf
from sklearn.datasets import load_iris

tf.reset_default_graph()

iris = load_iris()


def model_fn_(features, labels, mode, params):
    with tf.variable_scope("scope"):
        print()
        dense_layer = tf.layers.Dense(units=3, activation=None,
                                      kernel_initializer=tf.truncated_normal_initializer)

    y = dense_layer(features)
    probas = tf.nn.softmax(y)

    with tf.Session() as sess:
        print(sess)

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
    train_spec = tf.estimator.TrainSpec(input_fn=input_fn_,
                                        max_steps=500)
    eval_spec = tf.estimator.EvalSpec(input_fn=input_fn_, steps=5, start_delay_secs=5, throttle_secs=5)

    # evaluate做的确实会重新执行model_fn_但是执行之后，还会重新从model_dir参数中把模型数据load进来，然后进行评估
    tf.estimator.train_and_evaluate(estimator=model, train_spec=train_spec, eval_spec=eval_spec)


if __name__ == '__main__':
    # main()

    print("done")
