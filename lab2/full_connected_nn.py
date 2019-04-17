import tensorflow as tf
from lab1 import data_shuffle

DISPLAY_FREQUENCY = 10

tensor_board_path = 'D:/Programming/bsuir/sem4/MO/lab2/tensorboard'


class FullConnectedNN:
    def __init__(self, data, in_size, out_size, epochs):
        self.data = data

        self.in_size = in_size
        self.out_size = out_size

        self.epochs = epochs

    def build_model(self):
        self.x = tf.placeholder(tf.float32, shape=[None, self.in_size], name='IN')
        self.y = tf.placeholder(tf.float32, shape=[None, self.out_size], name='OUT')
        self.drop_rate = tf.placeholder(tf.float32)

        layer1 = full_connected_layer(self.x, 300, 'HFC1')
        layer2 = full_connected_layer(layer1, 300, 'HFC2')
        layer3 = full_connected_layer(layer2, 300, 'HFC3')

        self.logits = full_connected_layer(layer3, self.out_size, 'OFC1', output=True)

        l2_reg = 0
        loss1 = tf.nn.l2_loss(get_tensor_by_name('W_HFC1')) * l2_reg
        loss2 = tf.nn.l2_loss(get_tensor_by_name('W_HFC2')) * l2_reg
        loss3 = tf.nn.l2_loss(get_tensor_by_name('W_HFC3')) * l2_reg

        regularizer = loss1 + loss2 + loss3

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y, logits=self.logits))
        self.loss = loss + regularizer

        self.global_step = tf.Variable(0, trainable=False)
        learn_rate = tf.train.exponential_decay(0.05, self.global_step, len(self.data[0][0]), 1, staircase=True)

        self.optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=learn_rate
        ).minimize(
            loss=self.loss,
            global_step=self.global_step
        )

        self.correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

    def fit_model(self):
        batch_size = 128
        global_step = 0

        (x_train, y_train), (x_valid, y_valid) = self.data[:-1]

        num_train_iter = int(len(y_train) / batch_size)
        num_valid_iter = int(len(y_valid) / batch_size)

        best_acc = 0

        self.init = tf.global_variables_initializer()
        self.session = tf.InteractiveSession()
        self.session.run(self.init)

        tf.summary.scalar('accuracy', self.accuracy)
        tf.summary.scalar('loss', self.loss)

        self.merged = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(tensor_board_path + '/train', self.session.graph)
        self.valid_writer = tf.summary.FileWriter(tensor_board_path + '/valid')

        tf.global_variables_initializer().run()

        for epoch in range(self.epochs):
            print('Training epoch: {}'.format(epoch + 1))

            tr_x, tr_y = data_shuffle(x_train, y_train)
            v_x, v_y = data_shuffle(x_valid, y_valid)

            for iteration in range(num_train_iter):
                global_step += 1
                start = iteration * batch_size
                end = (iteration + 1) * batch_size
                x_batch, y_batch = get_next_batch(tr_x, tr_y, start, end)

                feed_dict_batch = {
                    self.x: x_batch,
                    self.y: y_batch,
                    self.drop_rate: 0
                }

                _, summary = self.session.run([self.optimizer, self.merged], feed_dict=feed_dict_batch)
                self.train_writer.add_summary(summary, global_step)

                if iteration % DISPLAY_FREQUENCY == 0:
                    loss_batch, acc_batch = self.session.run([self.loss, self.accuracy], feed_dict=feed_dict_batch)

                    print("iter {0:3d}:\t TRAIN Loss={1:.2f},\t Accuracy={2:.01%}".format(iteration, loss_batch,
                                                                                          acc_batch))

            acc_valid = 0
            for iteration in range(num_valid_iter):
                start = iteration * batch_size
                end = (iteration + 1) * batch_size
                x_batch, y_batch = get_next_batch(v_x, v_y, start, end)

                feed_dict_batch = {
                    self.x: x_batch,
                    self.y: y_batch,
                    self.drop_rate: 0
                }

                acc, summary = self.session.run([self.accuracy, self.merged], feed_dict=feed_dict_batch)
                acc_valid += acc

                self.valid_writer.add_summary(summary, global_step)

            acc_valid = acc_valid / num_valid_iter

            if acc_valid > best_acc:
                best_acc = acc_valid

            print('---------------------------------------------------------')
            print("Epoch: {0}:\t VALID Accuracy={1:.01%}".format(epoch + 1, acc_valid))
            print('---------------------------------------------------------')

    def test_model(self):
        (x_test, y_test) = self.data[2]

        batch_size = 256
        num_test_iter = int(len(x_test) / batch_size)

        acc_valid = 0
        global_step = 0
        for iteration in range(num_test_iter):
            global_step += 1
            start = iteration * batch_size
            end = (iteration + 1) * batch_size
            x_batch, y_batch = get_next_batch(x_test, y_test, start, end)

            feed_dict_batch = {
                self.x: x_batch,
                self.y: y_batch,
                self.drop_rate: 0
            }

            acc = self.session.run([self.accuracy], feed_dict=feed_dict_batch)
            acc_valid += acc

        print('---------------------------------------------------------')
        print("TEST Accuracy={0:.01%}".format(acc_valid / num_test_iter))
        print('---------------------------------------------------------')


def full_connected_layer(x, num_units, name, output=False, dropout=0):
    in_dim = x.get_shape()[1]

    W = weight_variable(name, shape=[in_dim, num_units])
    b = bias_variable(name, [num_units])

    layer = tf.matmul(x, W) + b

    if not output:
        layer = tf.nn.relu(layer)

    layer = tf.nn.dropout(layer, rate=dropout)

    return layer


def get_tensor_by_name(name):
    return tf.get_default_graph().get_tensor_by_name(f'{name}:0')


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


def get_next_batch(data_x, data_y, start, end):
    x_batch = data_x[start:end]
    y_batch = data_y[start:end]

    return x_batch, y_batch
