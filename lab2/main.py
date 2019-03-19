import numpy as np
import tensorflow as tf
from lab1 import small_data, split_data, get_unique_data, encode_classes, generate_one_hot_encoded_class, data_shuffle

from lab2.grid_search import GridSearch

HP_BATCH_SIZE = 'batch_size'
HP_LEARNING_RATE = 'learning_rate'
HP_UNITS = 'units'
HP_L1_SCALE = 'l1_scale'
HP_L2_SCALE = 'l2_scale'
HP_DROPOUT_RATE = 'dropout_rate'
HP_DECAY_RATE = 'decay_rate'
HP_EPOCHS = 'epochs'
HP_ACT_FUNC = 'act_func'

RND_SEED = None
tf.random.set_random_seed(RND_SEED)
np.random.seed(RND_SEED)

DISPLAY_FREQUENCY = 2000

# Img props
img_w = 28
img_h = 28
img_size = img_h * img_w
img_classes = 10

# # Learning hyper params
# epochs = 100
# batch_size = 100
# learning_rate = 0.001
# units = [300 for u in range(2)]


# # L1 Regularization
use_l1_regularization = True
# l1_scale = 0.001


# # L2 Regularization
use_l2_regularization = True
# l2_scale = 0.001


# # Dropout
use_dropout = True
# dropout_rate = 0.15


# # Adaptive lr
use_adaptive_lr = True
# decay_rate = 0.95


def prepare_data(path_to_data):
    one_hot_encoded_labels = generate_one_hot_encoded_class()

    images, labels = get_unique_data(path_to_data)
    labels = encode_classes(labels, one_hot_encoded_labels)

    tr_x, tr_y, v_x, v_y, te_x, te_y = split_data(images, labels, 0.8, 0.1, 0.1)
    tr_x = tr_x.reshape(tr_x.shape[0], -1)
    v_x = v_x.reshape(v_x.shape[0], -1)
    te_x = te_x.reshape(te_x.shape[0], -1)

    return tr_x, tr_y, v_x, v_y, te_x, te_y


d_tr_x, d_tr_y, d_v_x, d_v_y, d_te_x, d_te_y = prepare_data(small_data)


def main():
    grid_search = GridSearch(
        {
            HP_BATCH_SIZE: np.linspace(50, 500, 7, dtype='int16'),
            HP_LEARNING_RATE: np.linspace(0.001, 0.01, 7),
            HP_UNITS: [
                *[[x] * y for x in [50, 100, 300, 500, 700] for y in range(1, 6)],
                [50, 100, 200, 300, 500],
                [500, 300, 200, 100, 50]
            ],
            HP_L1_SCALE: np.linspace(0, 0.3, 7),
            HP_L2_SCALE: np.linspace(0, 0.3, 7),
            HP_DROPOUT_RATE: np.linspace(0.05, 0.2, 7),
            HP_DECAY_RATE: np.linspace(1, 0.9, 7),
            HP_ACT_FUNC: [tf.nn.relu, tf.nn.sigmoid, tf.nn.tanh, tf.nn.elu],
            HP_EPOCHS: [70]
        },
        fit_model,
        'accuracy',
        100
    )
    # grid_search = GridSearch(
    #     {
    #         HP_BATCH_SIZE: np.linspace(50, 500, 2, dtype='int32'),
    #         HP_LEARNING_RATE: np.linspace(0.001, 0.01, 2),
    #         HP_UNITS: [
    #             *[[x] * y for x in [50, 100, 300, 500, 700] for y in range(1, 3)],
    #             [50, 100, 200, 300, 500],
    #             [500, 300, 200, 100, 50]
    #         ],
    #         HP_L1_SCALE: np.linspace(0, 0.3, 2),
    #         HP_L2_SCALE: np.linspace(0, 0.3, 2),
    #         HP_DROPOUT_RATE: np.linspace(0.05, 0.2, 2),
    #         HP_DECAY_RATE: np.linspace(1, 0.9, 2),
    #         HP_ACT_FUNC: [tf.nn.relu, tf.nn.sigmoid, tf.nn.tanh, tf.nn.elu],
    #         HP_EPOCHS: [50]
    #     },
    #     fit_model,
    #     'accuracy',
    #     1
    # )

    grid_search.execute()

    print(grid_search.get_results())
    y = 0
    # fit_model(x, y, tr_x, tr_y, v_x, v_y, te_x, te_y, {
    #     HP_BATCH_SIZE: 100,
    #     HP_LEARNING_RATE: 0.001,
    #     HP_UNITS: [300, 300],
    #     HP_L1_SCALE: 0.001,
    #     HP_L2_SCALE: 0.001,
    #     HP_DROPOUT_RATE: 0.15,
    #     HP_DECAY_RATE: 0.95,
    #     HP_EPOCHS: 50,
    #     HP_ACT_FUNC: tf.nn.relu
    # })


def fit_model(hp):
    in_x = tf.placeholder(tf.float32, shape=[None, img_size], name='X')
    out_y = tf.placeholder(tf.float32, shape=[None, img_classes], name='Y')

    init, loss_function, optimizer, accuracy = init_nn(
        in_x,
        out_y,
        len(d_tr_x),
        hp[HP_UNITS],
        hp[HP_DROPOUT_RATE],
        hp[HP_L1_SCALE],
        hp[HP_L2_SCALE],
        hp[HP_LEARNING_RATE],
        hp[HP_DECAY_RATE],
        hp[HP_ACT_FUNC]
    )

    sess = tf.InteractiveSession()
    sess.run(init)

    best_score = train(sess, in_x, out_y, optimizer, loss_function, accuracy, hp[HP_EPOCHS], hp[HP_BATCH_SIZE])
    tf.reset_default_graph()
    # test_nn(sess, loss_function, accuracy)
    return best_score


def init_nn(x, y, train_size, units, droput_rate, l1_scale, l2_scale, learn_rate, decay_rate, act_func):
    output_logits = init_layers(x, units, droput_rate, act_func)

    loss_function = init_loss(y, output_logits, units, l1_scale, l2_scale)

    learn_rate, global_step = init_learning_rate(train_size, learn_rate, decay_rate)

    optimizer = tf.train.GradientDescentOptimizer(
        learning_rate=learn_rate
    ).minimize(
        loss=loss_function,
        global_step=global_step
    )

    correct_prediction = tf.equal(tf.argmax(output_logits, 1), tf.argmax(y, 1), name='correct_prediction')
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
    init = tf.global_variables_initializer()

    return init, loss_function, optimizer, accuracy


def init_learning_rate(train_size, learning_rate, decay_rate):
    global_step = None
    learn_rate = learning_rate

    if use_adaptive_lr:
        global_step = tf.Variable(0, trainable=False)

        learn_rate = tf.train.exponential_decay(
            learning_rate,
            global_step,
            train_size,
            decay_rate,
            staircase=True
        )

    return learn_rate, global_step


def init_layers(x, units, dropout_rate, act_func):
    last_layout = x
    for hl in range(len(units)):
        last_layout = full_connected_layer(
            last_layout,
            units[hl],
            f'HFCL{hl}',
            act_func,
            use_dropout,
            dropout_rate,
            output=True
        )

    return full_connected_layer(last_layout, img_classes, 'OFCL', None, False, None, output=True)


def l1_loss(w):
    return tf.reduce_sum(tf.abs(w))


def init_loss(y, logits, units, l1_scale, l2_scale):
    loss_function = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits), name='loss_function'
    )
    l1_penalty = 0
    l2_penalty = 0

    weights = []
    for i in range(len(units)):
        weights.append(get_tensor_by_name(f'W_HFCL{i}'))
    weights.append(get_tensor_by_name('W_OFCL'))

    if use_l1_regularization:
        wrap_loss = np.vectorize(lambda a: l1_loss(a))
        l1_penalty = l1_scale * np.sum(wrap_loss(weights), axis=0)

    if use_l2_regularization:
        wrap_loss = np.vectorize(lambda a: tf.nn.l2_loss(a))
        l2_penalty = l2_scale * np.sum(wrap_loss(weights), axis=0)

    return loss_function + l1_penalty + l2_penalty


def get_tensor_by_name(name):
    return tf.get_default_graph().get_tensor_by_name(f'{name}:0')


def train(sess, x, y, optimizer, loss_function, accuracy, epochs, batch_size):
    global_step = 0
    num_tr_iter = int(len(d_tr_y) / batch_size)
    best_score = {
        'accuracy': 0,
        'loss': 0
    }

    for epoch in range(epochs):
        print('Training epoch: {}'.format(epoch + 1))

        tr_x, tr_y = data_shuffle(d_tr_x, d_tr_y)

        for iteration in range(num_tr_iter):
            global_step += 1
            start = iteration * batch_size
            end = (iteration + 1) * batch_size
            x_batch, y_batch = get_next_batch(tr_x, tr_y, start, end)

            feed_dict_batch = {
                x: x_batch,
                y: y_batch
            }

            sess.run(optimizer, feed_dict=feed_dict_batch)

            if iteration % DISPLAY_FREQUENCY == 0:
                loss_batch, acc_batch = sess.run([loss_function, accuracy], feed_dict=feed_dict_batch)

                print("iter {0:3d}:\t TRAIN Loss={1:.2f},\t Accuracy={2:.01%}".format(iteration, loss_batch, acc_batch))

        feed_dict_valid = {
            x: d_v_x,
            y: d_v_y
        }

        loss_valid, acc_valid = sess.run([loss_function, accuracy], feed_dict=feed_dict_valid)

        if acc_valid > best_score['accuracy']:
            best_score['accuracy'] = acc_valid
            best_score['loss'] = loss_valid

        print('---------------------------------------------------------')
        print("Epoch: {0}:\t VALID Loss={1:.2f},\t Accuracy={2:.01%}".format(epoch + 1, loss_valid, acc_valid))
        print('---------------------------------------------------------')

    return best_score


def test_nn(sess, x, y, loss_function, accuracy):
    feed_dict_valid = {
        x: d_te_x,
        y: d_te_y
    }

    loss_valid, acc_valid = sess.run([loss_function, accuracy], feed_dict=feed_dict_valid)

    print('---------------------------------------------------------')
    print("TEST Loss: {0:.2f}, Accuracy: {1:.01%}".format(loss_valid, acc_valid))
    print('---------------------------------------------------------')


def weight_variable(name, shape):
    initial = tf.truncated_normal_initializer(stddev=0.01)

    return tf.get_variable('W_' + name,
                           dtype=tf.float32,
                           shape=shape,
                           initializer=initial)


def bias_variable(name, shape):
    initial = tf.constant(0., shape=shape, dtype=tf.float32)

    return tf.get_variable('b_' + name,
                           dtype=tf.float32,
                           initializer=initial)


def full_connected_layer(x, num_units, name, act_func, drop_out, dropout_rate, output=False):
    in_dim = x.get_shape()[1]

    W = weight_variable(name, shape=[in_dim, num_units])
    b = bias_variable(name, [num_units])

    layer = tf.matmul(x, W) + b

    if not output:
        layer = act_func(layer)

    if drop_out:
        layer = tf.nn.dropout(layer, rate=dropout_rate)

    return layer


def get_next_batch(data_x, data_y, start, end):
    x_batch = data_x[start:end]
    y_batch = data_y[start:end]

    return x_batch, y_batch


if __name__ == '__main__':
    main()
