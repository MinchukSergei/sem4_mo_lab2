import numpy as np
import tensorflow as tf
from lab1 import small_data, split_data, get_unique_data, encode_classes, generate_one_hot_encoded_class, data_shuffle

from lab2.grid_search import GridSearch

RND_SEED = 42

# Img props
img_w = 28
img_h = 28
img_size = img_h * img_w
img_classes = 10

# Learning hyper params
epochs = 50
batch_size = 100
display_freq = 70
learning_rate = 0.001
units = [300 for u in range(2)]
# units = [u for u in range(2500, 0, -500)]
# units = [300, 500, 250, 125, 63]

# L2 Regularization
use_regularization = True
l2_beta = 0.01

# Dropout
use_dropout = True
dropout_rate = 0.15

# Adaptive lr
use_adaptive_lr = True
start_learning_rate = 0.005
decay_rate = 0.95


def main():
    tf.random.set_random_seed(RND_SEED)
    np.random.seed(RND_SEED)

    tr_x, tr_y, v_x, v_y, te_x, te_y = prepare_data(small_data)

    x = tf.placeholder(tf.float32, shape=[None, img_size], name='X')
    y = tf.placeholder(tf.float32, shape=[None, img_classes], name='Y')

    init, loss_function, optimizer, accuracy = init_nn(x, y, len(tr_x))

    sess = train(init, x, y, optimizer, loss_function, accuracy, tr_x, tr_y, v_x, v_y)

    test_nn(x, y, te_x, te_y, sess, loss_function, accuracy)


def prepare_data(path_to_data):
    one_hot_encoded_labels = generate_one_hot_encoded_class()

    images, labels = get_unique_data(path_to_data)
    labels = encode_classes(labels, one_hot_encoded_labels)

    tr_x, tr_y, v_x, v_y, te_x, te_y = split_data(images, labels, 0.8, 0.1, 0.1)
    tr_x = tr_x.reshape(tr_x.shape[0], -1)
    v_x = v_x.reshape(v_x.shape[0], -1)
    te_x = te_x.reshape(te_x.shape[0], -1)

    return tr_x, tr_y, v_x, v_y, te_x, te_y


def init_nn(x, y, train_size):
    output_logits = init_layers(x)

    loss_function = init_loss(y, output_logits)

    learn_rate, global_step = init_learning_rate(train_size)

    optimizer = tf.train.GradientDescentOptimizer(
        learning_rate=learn_rate
    ).minimize(
        loss=loss_function,
        global_step=global_step
    )
    # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name='Adam-Optimizer').minimize(loss_function)

    correct_prediction = tf.equal(tf.argmax(output_logits, 1), tf.argmax(y, 1), name='correct_prediction')
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
    init = tf.global_variables_initializer()

    return init, loss_function, optimizer, accuracy


def init_learning_rate(train_size):
    global_step = None
    learn_rate = learning_rate

    if use_adaptive_lr:
        global_step = tf.Variable(0, trainable=False)

        learn_rate = tf.train.exponential_decay(
            start_learning_rate,
            global_step,
            train_size,
            decay_rate,
            staircase=True
        )

    return learn_rate, global_step


def init_layers(x):
    last_layout = x
    for hl in range(len(units)):
        last_layout = full_connected_layer(last_layout, units[hl], f'HFCL{hl}', drop_out=use_dropout)

    return full_connected_layer(last_layout, img_classes, 'OFCL', drop_out=False)


def init_loss(y, logits):
    loss_function = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits), name='loss_function'
    )

    if use_regularization:
        regularizers = []

        for i in range(len(units)):
            regularizers.append(tf.nn.l2_loss(get_tensor_by_name(f'W_HFCL{i}')))
        regularizers.append(tf.nn.l2_loss(get_tensor_by_name('W_OFCL')))

        loss_function = tf.reduce_mean(loss_function + l2_beta * np.sum(regularizers, axis=0))

    return loss_function


def get_tensor_by_name(name):
    return tf.get_default_graph().get_tensor_by_name(f'{name}:0')


def train(init, x, y, optimizer, loss_function, accuracy, tr_x, tr_y, v_x, v_y):
    sess = tf.InteractiveSession()
    sess.run(init)
    global_step = 0
    num_tr_iter = int(len(tr_y) / batch_size)

    for epoch in range(epochs):
        print('Training epoch: {}'.format(epoch + 1))

        tr_x, tr_y = data_shuffle(tr_x, tr_y)

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

            if iteration % display_freq == 0:
                loss_batch, acc_batch = sess.run([loss_function, accuracy], feed_dict=feed_dict_batch)

                print("iter {0:3d}:\t TRAIN Loss={1:.2f},\t Accuracy={2:.01%}".format(iteration, loss_batch, acc_batch))

        feed_dict_valid = {
            x: v_x,
            y: v_y
        }

        loss_valid, acc_valid = sess.run([loss_function, accuracy], feed_dict=feed_dict_valid)

        print('---------------------------------------------------------')
        print("Epoch: {0}:\t VALID Loss={1:.2f},\t Accuracy={2:.01%}".format(epoch + 1, loss_valid, acc_valid))
        print('---------------------------------------------------------')

    return sess


def test_nn(x, y, te_x, te_y, sess, loss_function, accuracy):
    feed_dict_valid = {
        x: te_x,
        y: te_y
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


def full_connected_layer(x, num_units, name, drop_out):
    in_dim = x.get_shape()[1]

    W = weight_variable(name, shape=[in_dim, num_units])
    b = bias_variable(name, [num_units])

    layer = tf.matmul(x, W) + b

    layer = tf.nn.relu(layer)

    if drop_out:
        layer = tf.nn.dropout(layer, rate=dropout_rate)

    return layer


def get_next_batch(x, y, start, end):
    x_batch = x[start:end]
    y_batch = y[start:end]

    return x_batch, y_batch


if __name__ == '__main__':
    main()
