from lab1 import small_data, split_data, get_unique_data, encode_classes, generate_one_hot_encoded_class

from lab2.full_connected_nn import FullConnectedNN

# Img props
img_w = 28
img_h = 28
img_size = img_h * img_w
img_classes = 10

# # Learning hyper params
epochs = 100
batch_size = 100
learning_rate = 0.001
units = [300, 300]

# # L1 Regularization
use_l1_regularization = True
l1_scale = 0.001

# # L2 Regularization
use_l2_regularization = True
l2_scale = 0.001

# # Dropout
use_dropout = True
dropout_rate = 0.15

# # Adaptive lr
use_adaptive_lr = True
decay_rate = 0.95


def main():
    fc_nn = FullConnectedNN(
        prepare_data(small_data),
        img_size,
        img_classes,
        epochs=100
    )

    fc_nn.build_model()
    fc_nn.fit_model()
    fc_nn.test_model()


def prepare_data(path_to_data):
    one_hot_encoded_labels = generate_one_hot_encoded_class()

    images, labels = get_unique_data(path_to_data)
    labels = encode_classes(labels, one_hot_encoded_labels)

    x_train, y_train, x_valid, y_valid, x_test, y_test = split_data(images, labels, 0.8, 0.1, 0.1)

    x_train = x_train.reshape(x_train.shape[0], -1)
    x_valid = x_valid.reshape(x_valid.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)

    return (x_train / 255, y_train), (x_valid / 255, y_valid), (x_test / 255, y_test)


if __name__ == '__main__':
    main()
