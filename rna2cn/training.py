import matplotlib
matplotlib.use('Agg')

import keras
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import sklearn.metrics
import os
import re
import glob
import logging
import pandas as pd
import numpy as np
import seaborn as sns
import pickle
import time
import argparse
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K
from keras.models import Sequential, model_from_json
from keras.layers import LSTM, Dense, Masking, Dropout, Conv1D, Activation, Input, Permute, Reshape
from keras.layers.wrappers import Bidirectional, TimeDistributed

import seaborn as sns
import matplotlib.pyplot as plt


sns.set_style('whitegrid')
chromosomes = list(map(str, range(1, 23)))  # + ['X']


def getargs(argv):
    parser = argparse.ArgumentParser(prog='RNA2CN train')
    parser.add_argument('--data', required=True,
                        help='Input data pickle')
    parser.add_argument('--epochs', required=True, type=int,
                        help='Number of epochs to train for')
    parser.add_argument('--batch-size', required=True, type=int,
                        help='Batch size')
    parser.add_argument('--model', required=True,
                        help='Serialised model json to train', default=None)
    parser.add_argument('--output', required=True,
                        help='Folder to dump jsons of trained models to', default=None)
    parser.add_argument('--lr', required=False, type=float,
                        help='RMSProp learning rate', default=0.01)
    return parser.parse_args(argv)


def make_batches(size, batch_size):
    """Returns a list of batch indices (tuples of indices).
    # Arguments
        size: Integer, total size of the data to slice into batches.
        batch_size: Integer, batch size.
    # Returns
        A list of tuples of array indices.
    """
    indices = shuffle(np.arange(size))
    num_batches = int(size / batch_size)
    rem = size % batch_size

    batches = np.split(indices[:size - rem], num_batches)
    return batches + [np.concatenate((indices[-rem:], indices[:(batch_size - rem)]))]


def evaluate(model, X, Y, mask, chr_steps, bs):
    '''Collapses cells and evaluates predictions from X against truth Y,
         returns loss, accuracy, MSE. Loss isn't implemented'''
    chroffset = [0] + list(np.cumsum(chr_steps))
    pred = predict(model, X, chr_steps, bs)
    yhat = pred.argmax(axis=-1)[:, mask].reshape((len(X), -1))
    Y = Y.argmax(axis=-1)[:, mask].reshape((len(X), -1))
    acc = sklearn.metrics.accuracy_score(Y.ravel(), yhat.ravel())
    mse = sklearn.metrics.mean_squared_error(Y.ravel(), yhat.ravel())
    return np.array([0, acc, mse])


def predict(model, X, chr_steps, bs, n_out=6):
    '''Run model and get predictions for an input X'''
    chroffset = [0] + list(np.cumsum(chr_steps))
    pred = np.zeros((*X.shape[:-1], n_out))
    for batch in make_batches(len(X), bs):
        for i, chr in list(enumerate(chromosomes)):
            model.reset_states()
            for j in range(chr_steps[i]):
                x = X[batch, chroffset[i] + j]
                pred[batch, chroffset[i] + j] = model.predict(x, verbose=0, batch_size=bs)
    return pred


def makedata(X, Y, train_cells):
    '''Split X,Y cellwise into train and test sets,
       returns (X_train, Y_train, X_test, Y_test, test_cells)'''
    test_cells = np.array([x for x in range(len(X)) if x not in train_cells])

    X_train, Y_train = X[train_cells], Y[train_cells]
    X_test, Y_test = X[test_cells], Y[test_cells]

    return X_train, Y_train, X_test, Y_test, test_cells


def plotLearning(model, fout):
    '''Draws the learning curve for model
       up to the current epoch to path fout'''
    plt.figure(figsize=(16, 6))

    plt.subplot(121).set_title("Error")
    plt.subplot(121).plot(1 - np.array(model.history)[:, 4], label='Test error')
    plt.subplot(121).plot(1 - np.array(model.history)[:, 1], label='Training error')
    plt.subplot(121).set_xlabel('Epoch')
    plt.legend()

    plt.subplot(122).set_title("MSE")
    plt.subplot(121).set_xlabel('Epoch')
    plt.subplot(122).plot(np.array(model.history)[:, 5], label='Test MSE')
    plt.subplot(122).plot(np.array(model.history)[:, 2], label='Train MSE')
    plt.legend()

    plt.savefig(fout)
    plt.close()


def train(model, n_epochs, X_train, Y_train, X_test, Y_test, mask, chr_steps, out_dir, bs):
    '''Train the model for n_epochs on the provided data.
       Saves hd5s of the model weights to out_dir and updates
       the learning curve plot every 10 epochs'''
    chroffset = [0] + list(np.cumsum(chr_steps))
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    for epoch in range(n_epochs):
        epoch_start, epoch_hist = time.time(), []
        for batch in make_batches(len(X_train), bs):
            for i, chr in enumerate(shuffle(chromosomes)):
                for j in range(chr_steps[i]):
                    x = X_train[batch, chroffset[i] + j]
                    y = Y_train[batch, chroffset[i] + j]
                    epoch_hist.append(model.train_on_batch(x, y))
                model.reset_states()

        model.history.append(np.concatenate([np.mean(epoch_hist, axis=0), evaluate(model, X_test, Y_test, mask, chr_steps, bs)]))
        print("Epoch {0}: {1:.1f}s".format(len(model.history), time.time() - epoch_start))
        print('-'*20)
        print("Train Accuracy: {0:.2%}, Train MSE: {1:.2f}".format(model.history[-1][1], model.history[-1][2]))
        print("Test Accuracy: {0:.2%}, Test MSE: {1:.2f}\n".format(model.history[-1][4], model.history[-1][5]))

        if (epoch + 1) % 10 == 0:
            print("Saving model state to file " + os.path.join(out_dir, "epoch" + str(epoch + 1) + ".json"))
            model.save_weights(os.path.join(out_dir, "epoch" + str(epoch + 1) + ".hd5"))
            plotLearning(model, os.path.join(out_dir, "learningcurve.pdf"))
            with open(os.path.join(out_dir, "history.pickle"), 'wb') as f:
                pickle.dump(model.history, f)
    return model


def train_command(argv):

    args = getargs(argv)

    with open(args.data, 'rb') as f:
        X, Y, mask, train_cells, chr_steps, chr_boundaries, _ = pickle.load(f)

    with open(args.model, 'r') as f:
        model = model_from_json(f.read())
    print("Loaded model from file " + args.model)
    print(model.summary())

    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.RMSprop(lr=args.lr),
                  metrics=['accuracy', 'mse'])
    model.history = []

    *data, test_cells = makedata(X, Y, train_cells)
    train(model, args.epochs, *data, mask, chr_steps, out_dir=args.output, bs=args.batch_size)
