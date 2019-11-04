from __future__ import print_function

import os
import xd_data
import argparse
import densenet
import numpy as np
import keras.backend as K
from sklearn.metrics import classification_report

# from keras.datasets import cifar10
from keras.optimizers import Adam
from keras.utils import np_utils
from xd_data import session_creater


def run_cifar10(data_dir,
                batch_size,
                nb_epoch,
                depth,
                nb_dense_block,
                nb_filter,
                growth_rate,
                dropout_rate,
                learning_rate,
                weight_decay,
                plot_architecture):
    """ Run CIFAR10 experiments

    :param batch_size: int -- batch size
    :param nb_epoch: int -- number of training epochs
    :param depth: int -- network depth
    :param nb_dense_block: int -- number of dense blocks
    :param nb_filter: int -- initial number of conv filter
    :param growth_rate: int -- number of new filters added by conv layers
    :param dropout_rate: float -- dropout rate
    :param learning_rate: float -- learning rate
    :param weight_decay: float -- weight decay
    :param plot_architecture: bool -- whether to plot network architecture

    """

    ###################
    # Data processing #
    ###################

    # the data, shuffled and split between train and test sets
    # (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    (X_train, y_train), (X_test, y_test) = xd_data.load_data(data_dir)

    nb_classes = max(y_train.tolist() + y_test.tolist()) + 1
    img_dim = X_train.shape[1:]

    if K.image_data_format() == "channels_first":
        n_channels = X_train.shape[1]
    else:
        n_channels = X_train.shape[-1]
    print("nb_classes: {}, img_dim: {}, n_channels: {}".format(
        nb_classes, img_dim, n_channels))

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    # Normalisation
    X = np.vstack((X_train, X_test))
    # 2 cases depending on the image ordering
    if K.image_data_format() == "channels_first":
        for i in range(n_channels):
            mean = np.mean(X[:, i, :, :])
            std = np.std(X[:, i, :, :])
            X_train[:, i, :, :] = (X_train[:, i, :, :] - mean) / std
            X_test[:, i, :, :] = (X_test[:, i, :, :] - mean) / std

    elif K.image_data_format() == "channels_last":
        for i in range(n_channels):
            mean = np.mean(X[:, :, :, i])
            std = np.std(X[:, :, :, i])
            X_train[:, :, :, i] = (X_train[:, :, :, i] - mean) / std
            X_test[:, :, :, i] = (X_test[:, :, :, i] - mean) / std

    ###################
    # Construct model #
    ###################

    model = densenet.DenseNet(nb_classes,
                              img_dim,
                              depth,
                              nb_dense_block,
                              growth_rate,
                              nb_filter,
                              dropout_rate=dropout_rate,
                              weight_decay=weight_decay)
    # Model output
    model.summary()

    # Build optimizer
    opt = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=["accuracy"])

    if plot_architecture:
        from keras.utils.visualize_util import plot
        plot(model, to_file='./figures/densenet_archi.png', show_shapes=True)

    ####################
    # Network training #
    ####################

    print("Training")
    with session_creater():
      model.fit(x=X_train,
                y=Y_train,
                epochs=nb_epoch,
                batch_size=batch_size,
                validation_split=0.1)
      preds = model.predict(X_test)
      pred_labels = np.argmax(preds, 1)
      report = classification_report(y_test, pred_labels)
      print(report)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run CIFAR10 experiment')
    parser.add_argument('--data_dir', default="./xd-data", type=str,
                        help='data directory')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size')
    parser.add_argument('--nb_epoch', default=30, type=int,
                        help='Number of epochs')
    parser.add_argument('--depth', type=int, default=7,
                        help='Network depth')
    parser.add_argument('--nb_dense_block', type=int, default=1,
                        help='Number of dense blocks')
    parser.add_argument('--nb_filter', type=int, default=16,
                        help='Initial number of conv filters')
    parser.add_argument('--growth_rate', type=int, default=12,
                        help='Number of new filters added by conv layers')
    parser.add_argument('--dropout_rate', type=float, default=0.2,
                        help='Dropout rate')
    parser.add_argument('--learning_rate', type=float, default=1E-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1E-4,
                        help='L2 regularization on weights')
    parser.add_argument('--plot_architecture', type=bool, default=False,
                        help='Save a plot of the network architecture')

    args = parser.parse_args()

    print("Network configuration:")
    for name, value in parser.parse_args()._get_kwargs():
        print(name, value)

    list_dir = ["./log", "./figures"]
    for d in list_dir:
        if not os.path.exists(d):
            os.makedirs(d)

    run_cifar10(args.data_dir,
                args.batch_size,
                args.nb_epoch,
                args.depth,
                args.nb_dense_block,
                args.nb_filter,
                args.growth_rate,
                args.dropout_rate,
                args.learning_rate,
                args.weight_decay,
                args.plot_architecture)
