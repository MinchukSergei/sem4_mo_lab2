import numpy as np
import tensorflow as tf
from lab1 import small_data, split_data, get_unique_data, encode_classes, generate_one_hot_encoded_class

img_w = 28
img_h = 28
img_size = img_h * img_w
img_classes = 10

epochs = 100
batch_size = 100
display_freq = 70
learning_rate = 0.001

number_of_layers = 3
number_of_units = 300

rnd_seed = 42
l2_beta = 0.01

dropout = 0.15


def main():
    one_hot_encoded_labels = generate_one_hot_encoded_class()

    images, labels = get_unique_data(small_data)
    labels = encode_classes(labels, one_hot_encoded_labels)

    tr_x, tr_y, te_x, te_y, v_x, v_y = split_data(images, labels, 0.7, 0.15, 0.15, rnd_seed)
    tr_x = tr_x.reshape(tr_x.shape[0], -1)
    te_x = te_x.reshape(te_x.shape[0], -1)
    v_x = v_x.reshape(v_x.shape[0], -1)

    x = tf.placeholder(tf.float32, shape=[None, img_size], name='X')
    y = tf.placeholder(tf.float32, shape=[None, img_classes], name='Y')

    init, class_prediction, loss_function, optimizer, accuracy = init_nn(x, y, number_of_layers)
    sess = train(init, x, y, optimizer, loss_function, accuracy, tr_x, tr_y, v_x, v_y, rnd_seed)
    test_nn(x, y, te_x, te_y, sess, loss_function, accuracy)
    pass


def init_nn(x, y, hidden_layouts):
    regularizers = []

    fc_layer_in, W_IN = full_connected_layer(x, number_of_units, 'IN', True)
    regularizers.append(W_IN)

    last_layout = fc_layer_in
    for hl in range(hidden_layouts):
        last_layout, W = full_connected_layer(last_layout, number_of_units, f'full_connected_{hl}', True)
        regularizers.append(W)

    output_logits, W_OUT = full_connected_layer(last_layout, img_classes, 'OUT', False)
    regularizers.append(W_OUT)
    regularizers = np.array(regularizers)

    class_prediction = tf.argmax(output_logits, axis=1, name='predictions')
    loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=output_logits),
                                   name='loss_function')
    v_func = np.vectorize(lambda a: tf.nn.l2_loss(a))
    # loss_function = tf.reduce_mean(loss_function + l2_beta * np.sum(v_func(regularizers), axis=0))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss_function)
    # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name='Adam-Optimizer').minimize(loss_function)

    correct_prediction = tf.equal(tf.argmax(output_logits, 1), tf.argmax(y, 1), name='correct_prediction')
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
    init = tf.global_variables_initializer()

    return init, class_prediction, loss_function, optimizer, accuracy


def train(init, x, y, optimizer, loss_function, accuracy, tr_x, tr_y, v_x, v_y, rnd_seed):
    sess = tf.InteractiveSession()
    sess.run(init)
    global_step = 0
    num_tr_iter = int(len(tr_y) / batch_size)

    for epoch in range(epochs):
        print('Training epoch: {}'.format(epoch + 1))

        tr_x, tr_y = data_shuffle(tr_x, tr_y, rnd_seed)

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
        print("Epoch: {0}, VALIDATION Loss: {1:.2f}, Accuracy: {2:.01%}".format(epoch + 1, loss_valid, acc_valid))
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


def full_connected_layer(x, num_units, name, use_relu=True):
    in_dim = x.get_shape()[1]

    W = weight_variable(name, shape=[in_dim, num_units])
    b = bias_variable(name, [num_units])

    layer = tf.matmul(x, W)
    layer += b

    if use_relu:
        layer = tf.nn.relu(layer)
        # layer = tf.nn.dropout(layer, rate=dropout)

    return layer, W


def data_shuffle(x, y, seed):
    if seed is not None:
        np.random.seed(seed)

    permutation = np.random.permutation(y.shape[0])
    x_shuffled = x[permutation, :]
    y_shuffled = y[permutation, :]

    return x_shuffled, y_shuffled


def get_next_batch(x, y, start, end):
    x_batch = x[start:end]
    y_batch = y[start:end]

    return x_batch, y_batch


if __name__ == '__main__':
    main()
