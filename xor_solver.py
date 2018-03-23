import sys
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


def lstm_model(features, labels, mode, params):
    hidden_state_size = params['hidden_state_size']
    learning_rate = params['learning_rate']

    inputs = tf.cast(
        tf.reshape(features['X'], [-1, features['X'].shape[1], 1]),
        dtype=tf.float32)

    lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=hidden_state_size)
    lstm_outputs, state = tf.nn.dynamic_rnn(
        cell=lstm_cell,
        inputs=inputs,
        sequence_length=features['seq_len'],
        dtype=tf.float32)

    last_output_indexes = tf.stack(
        [tf.range(tf.shape(inputs)[0]), features['seq_len']-1], axis=1)

    logits = tf.layers.dense(
        inputs=tf.gather_nd(lstm_outputs, last_output_indexes),
        units=2)

    predictions = {
        'classes': tf.argmax(input=logits, axis=1),
        'probabilities': tf.nn.softmax(logits, name="softmax_tensor"),
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode, predictions=predictions)

    eval_metric_ops = {
        'accuracy': tf.metrics.accuracy(
            labels=labels, predictions=predictions['classes']),
    }

    tf.summary.scalar('accuracy', eval_metric_ops['accuracy'][1])

    loss = tf.losses.sparse_softmax_cross_entropy(
        labels=labels, logits=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate)

        train_op = optimizer.minimize(
            loss=loss, global_step=tf.train.get_global_step())

        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, train_op=train_op)

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def train_estimator(estimator, data, hooks=None):
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'X': data['X'], 'seq_len': data['seq_len']},
        y=data['y'],
        batch_size=128,
        num_epochs=20,
        shuffle=True)

    estimator.train(
        input_fn=train_input_fn,
        hooks=hooks)


def eval_estimator(estimator, data):
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'X': data['X'], 'seq_len': data['seq_len']},
        y=data['y'],
        num_epochs=1,
        shuffle=False)

    eval_results = estimator.evaluate(input_fn=eval_input_fn)
    print()
    print(eval_results)
    print()


if __name__ == '__main__':
    seq_filename = sys.argv[1]
    seq_lengths_filename = sys.argv[2]
    mode = sys.argv[3]
    model_dir = sys.argv[4]

    seq = np.load(seq_filename)
    seq_lengths = np.load(seq_lengths_filename)
    seq_lengths = 50 * np.ones(100000, dtype=np.int32)
    seq_parity = np.array(
        [seq[i,0:seq_lengths[i]].sum() % 2 for i in range(seq_lengths.size)])

    X_train, X_val, y_train, y_val, seq_len_train, seq_len_val = (
        train_test_split(seq, seq_parity, seq_lengths, shuffle=False))

    train = {
        'X': X_train,
        'seq_len': seq_len_train.astype(np.int32),
        'y': y_train,
    }

    val = {
        'X': X_val,
        'seq_len': seq_len_val.astype(np.int32),
        'y': y_val,
    }

    params = {
        'hidden_state_size': 32,
        'learning_rate': 0.001,
    }

    lstm_estimator = tf.estimator.Estimator(
        model_fn=lstm_model,
        model_dir=model_dir,
        params=params)

    if mode == 'train':
        train_estimator(lstm_estimator, train)

    if mode == 'eval':
        print('\nTrain results:')
        eval_estimator(lstm_estimator, train)
        print('\nValidation results:')
        eval_estimator(lstm_estimator, val)
