from sklearn.datasets import load_iris
import tensorflow as tf
import numpy as np
import six

tf.reset_default_graph()

iris = load_iris()


def _as_graph_element(obj):
    """Retrieves Graph element."""
    graph = tf.get_default_graph()
    if not isinstance(obj, six.string_types):
        if not hasattr(obj, "graph") or obj.graph != graph:
            raise ValueError("Passed %s should have graph attribute that is equal "
                             "to current graph %s." % (obj, graph))
        return obj
    if ":" in obj:
        element = graph.as_graph_element(obj)
    else:
        element = graph.as_graph_element(obj + ":0")
        # Check that there is no :1 (e.g. it's single output).
        try:
            graph.as_graph_element(obj + ":1")
        except (KeyError, ValueError):
            pass
        else:
            raise ValueError("Name %s is ambiguous, "
                             "as this `Operation` has multiple outputs "
                             "(at least 2)." % obj)
    return element


class MyHook(tf.train.SessionRunHook):
    def __init__(self, tensors, every_n_iter):
        self._tensors = tensors
        self._timer = tf.train.SecondOrStepTimer(every_steps=every_n_iter)
        self._tag_order = sorted(tensors.keys())
        self._log_at_end=True
        self._saved_info=[]

    def begin(self):
        self._timer.reset()
        self._iter_count = 0
        # Convert names to tensors if given
        self._current_tensors = {tag: _as_graph_element(tensor)
                                 for (tag, tensor) in self._tensors.items()}

    def before_run(self, run_context):  # pylint: disable=unused-argument
        # 这个run_context里的original_args是原本会被送入session.run的参数
        # 如果返回了SessionRunArgs，那么会将SessionRunArgs其中的参数添加到session.run里，从而在after_run的时候获取其值
        # 本例之中，使用的dict，即名字->tensor，那么after_run的run_values的results也会是一个dict

        # 使用timer来判断是否需要触发
        self._should_trigger = self._timer.should_trigger_for_step(self._iter_count)
        if self._should_trigger:
            # 当返回了SessionRunArgs的时候，才会在after_run的地方获取到值
            return tf.train.SessionRunArgs(self._current_tensors)
        else:
            return None

    def _log_tensors(self, tensor_values):
        tf.logging.info("fuck")
        original = np.get_printoptions()
        np.set_printoptions(suppress=True)
        elapsed_secs, _ = self._timer.update_last_triggered_step(self._iter_count)
        stats = []
        for tag in self._tag_order:
            stats.append("%s = %s" % (tag, tensor_values[tag]))
        if elapsed_secs is not None:
            tf.logging.info("%s (%.3f sec)", ", ".join(stats), elapsed_secs)
        else:
            tf.logging.info("%s", ", ".join(stats))
        np.set_printoptions(**original)

    def after_run(self, run_context, run_values):
        _ = run_context
        print(run_values) # 可以看到是每log_step步的时候，run_values才会有数值，这是因为在before_run返回了RunArgs
        if self._should_trigger:
            self._log_tensors(run_values.results)
            self._saved_info.append(run_values.results['y_'])

        # 使用这个log本身去记录迭代的次数，因为只有当迭代次数是every_n_iter的整数倍的时候，run_values才有值
        self._iter_count += 1

    def end(self, session):
        if self._log_at_end:
            values = session.run(self._current_tensors)
            self._log_tensors(values)


def model_fn_(features, labels, mode, params):
    dense_layer = tf.layers.Dense(units=3, activation=None,
                                  kernel_initializer=tf.truncated_normal_initializer,
                                  name="dense")

    logits = tf.identity(dense_layer(features), name='logits')

    probas = tf.nn.softmax(logits, name="probas")

    predictions = {"logits": logits, "proba": probas}
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    else:
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
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

    log_tensors = {"dense_kernel": "dense/kernel", "y_": "logits"}
    tf.logging.set_verbosity(tf.logging.INFO)
    hooks = [MyHook(tensors=log_tensors, every_n_iter=5)]
    model.train(input_fn=input_fn_, hooks=hooks, max_steps=20)

    print(hooks[0]._saved_info)


if __name__ == '__main__':
    main()
    print("done")
