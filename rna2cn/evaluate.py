import matplotlib
matplotlib.use('Agg')

import argparse
from keras.models import model_from_json
import sklearn.metrics
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import pickle
import numpy as np
import os

from rna2cn.training import predict

chromosomes = list(map(str, range(1, 23)))  # + ['X']
sns.set_style('whitegrid')


def getargs(argv):
    parser = argparse.ArgumentParser(prog='RNA2CN evaluate')
    parser.add_argument('--data', required=True,
                        help='Input data pickle')
    parser.add_argument('--model', required=True,
                        help='Serialised model json to train', default=None)
    parser.add_argument('--weights', required=True,
                        help='=Trained model weights', default=None)
    parser.add_argument('--output', required=True,
                        help='Folder to dump jsons of trained models to', default=None)
    parser.add_argument('--metrics', required=False, type=str,
                        help='Sample metrics to include',
                        default='accuracy,breakpoints,events,gainloss')
    parser.add_argument('--confusion', action='store_true',
                        help='Plot the confusion matrix')
    parser.add_argument('--precision-recall', action='store_true',
                        help='Plot the precision recall curve')
    parser.add_argument('--plot', action='store_true',
                        help='Plot the predicted copy number profiles')

    return parser.parse_args(argv)


def makedata(X, Y, train_cells):
    '''Split X,Y cellwise into train and test sets,
       returns (X_train, Y_train, X_test, Y_test, test_cells)'''
    test_cells = np.array([x for x in range(len(X)) if x not in train_cells])

    X_train, Y_train = X[train_cells], Y[train_cells]
    X_test, Y_test = X[test_cells], Y[test_cells]

    return X_train, Y_train, X_test, Y_test, test_cells


def evaluate_command(argv):
    args = getargs(argv)

    # Load the model structure from JSON
    with open(args.model, 'r') as f:
        model = model_from_json(f.read())
    bs = model.layers[0].output_shape[0]
    print("Loaded model from file " + args.model)
    print(model.summary())

    # Load the trained model weights
    model.load_weights(args.weights)
    print("Loaded weights from " + args.weights)

    # Load the raw data
    with open(args.data, 'rb') as f:
        X, Y_oh, mask, train_cells, chr_steps, chr_boundaries, _ = pickle.load(f)

    *_, test_cells = makedata(X, Y_oh, train_cells)

    n_cells, n_points, n_outputs = Y_oh.shape[0], Y_oh.shape[2], Y_oh.shape[-1]
    n_features = X.shape[-1]
    Y = Y_oh.argmax(axis=-1)[:, mask].reshape((n_cells, -1))  # Y comes in one hot categorical encoding

    # Model prediction probabilities
    pred = predict(model, X, chr_steps, bs, n_outputs)
    # CN call is the most likely state
    yhat = pred.argmax(axis=-1)[:, mask].reshape((n_cells, -1))

    # Probability of there being a CN != 2 event at this site
    event_Y = np.where(Y_oh[..., 2], 0, 1)[:, mask]
    event_p = 1 - pred[:, mask, 2].reshape((n_cells, -1))

    # Probability of a gain/loss event
    gain_p = (pred[:, mask, 3] + pred[:, mask, 4] + pred[:, mask, 5]).reshape((n_cells, -1))
    loss_p = (pred[:, mask, 0] + pred[:, mask, 1]).reshape((n_cells, -1))
    normal_p = 1 - gain_p - loss_p
    gl_p = np.stack([loss_p, normal_p, gain_p], axis=-1)
    gl_Y = np.ones_like(Y)
    gl_Y[Y > 2] = 2
    gl_Y[Y < 2] = 0

    metrics = {k: None for k in args.metrics.split(',')}

    if 'accuracy' in metrics.keys():
        metrics['accuracy'] = sklearn.metrics.accuracy_score(Y[test_cells].ravel(), yhat[test_cells].ravel())
        metrics['mse'] = sklearn.metrics.mean_squared_error(Y[test_cells].ravel(), yhat[test_cells].ravel())

    if 'events' in metrics.keys():
        metrics['auc'] = sklearn.metrics.roc_auc_score(event_Y[test_cells].ravel(), event_p[test_cells].ravel())
        metrics['event_accuracy'] = sklearn.metrics.accuracy_score(event_Y[test_cells].ravel(), (event_p[test_cells] > 0.5).ravel())

    if 'gainloss' in metrics.keys():
        metrics['gainloss'] = sklearn.metrics.accuracy_score(gl_Y[test_cells].ravel(), np.argmax(gl_p[test_cells], axis=-1).ravel())

    print(metrics)

    if args.confusion:
        # Plot the confusion matrix
        confusion = sklearn.metrics.confusion_matrix(Y.ravel(), yhat.ravel())
        confusion = (confusion / confusion.sum(axis=0))

        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        sns.heatmap(np.nan_to_num(confusion), cmap=cmap, vmax=1, vmin=0, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5},
                    annot=True)
        plt.xlabel("Known")
        plt.ylabel("Predicted")
        plt.savefig(os.path.join(args.output, 'confusion.pdf'))
        plt.close()

    if args.precision_recall:
        # Plot the binary precision recall curve
        precision, recall, thresholds = sklearn.metrics.precision_recall_curve(event_Y[test_cells].ravel(), event_p[test_cells].ravel())
        plt.plot(precision, recall)
        plt.plot([0, 1], [1, 0], '--', color='gray')
        plt.xlabel("Precision")
        plt.ylabel("Recall")
        plt.title("Event detection")
        plt.savefig(os.path.join(args.output, 'prec_rec.pdf'))
        plt.close()

    if args.plot:
        # Make the full output plot
        with PdfPages(os.path.join(args.output, 'profiles.pdf')) as pdf:
            for i in range(n_cells):

                plt.figure(figsize=(9, 5))
                gridspec = (19, 1)

                ########## Subplot 1 #############
                ax1 = plt.subplot2grid(gridspec, (0, 0), rowspan=1)

                for j in range(n_outputs):
                    ax1.fill_between(range(mask.sum()), 0.95, 1.05, where=Y[i] == j)

                ax1.set_ylim((0.95, 1.05))
                ax1.set_xlim((0, mask.sum()))
                ax1.set_ylabel("Y", rotation=0, va='center')
                ax1.get_yaxis().set_label_coords(-0.05, 0.5)
                ax1.tick_params(which='both', bottom='off', left='off', labelbottom='off', labelleft='off')
                ax1.spines['top'].set_visible(False)
                ax1.spines['right'].set_visible(False)
                ax1.spines['left'].set_visible(False)
                ax1.spines['bottom'].set_visible(False)
                plt.title('Cell {0} - {1} - Acc: {2:.2%}'.format(i, 'Train' if i in train_cells else 'Test',
                                                                 sklearn.metrics.accuracy_score(Y[i].ravel(), yhat[i].ravel())))
                ax1.grid()

                ########## Subplot 2 #############
                ax2 = plt.subplot2grid(gridspec, (1, 0), rowspan=6)
                for j in range(n_outputs):
                    ax2.plot(pred[i, mask, j].reshape((-1)), label=str(j))
                ax2.vlines(chr_boundaries, 0, 1, linestyle='dashed', color='gray')
                ax2.set_ylim((-0.05, 1.05))
                ax2.set_xlim((0, mask.sum()))
                ax2.set_ylabel("Pr(CN)")
                ax2.get_yaxis().set_label_coords(-0.05, 0.5)
                ax2.set_yticks([0, 0.5, 1])
                ax2.set_yticklabels(['0', '0.5', '1'])
                ax2.tick_params(which='both', bottom='off', labelbottom='off')
                ax2.grid(b=False, axis='x')

                ########## Subplot 3 #############
                ax3 = plt.subplot2grid(gridspec, (7, 0), rowspan=6)
                for j in range(n_outputs):
                    ax3.fill_between(range(mask.sum()), j-0.15, j+0.15, where=yhat[i] == j, alpha=0.75)
                ax3.plot(Y[i], '.', color='black', ms=2)
                #ax3.plot(yhat[i], '.', color='red', ms=1)
                ax3.set_ylabel("Copy number")
                ax3.get_yaxis().set_label_coords(-0.05, 0.5)
                ax3.vlines(chr_boundaries, 0, 5, linestyle='dashed', color='gray')
                ax3.set_ylim((-0.2, n_outputs - 0.8))
                ax3.set_xlim((0, mask.sum()))
                ax3.set_yticks([1, 2, 3, 4])
                ax3.set_yticklabels(['1', '2', '3', '4'])
                ax3.tick_params(which='both', bottom='off', labelbottom='off')
                ax3.grid(b=False, axis='x')

                ########## Subplot 4#############
                ax4 = plt.subplot2grid(gridspec, (13, 0), rowspan=6)
                ax4.plot(event_p[i], color='black')
                ax4.fill_between(range(mask.sum()), 0, 1, where=event_Y[i] == 1, alpha=0.25)
                ax4.vlines(chr_boundaries, 0, 1, linestyle='dashed', color='gray')
                ax4.set_ylabel("Pr(Event)")
                ax4.get_yaxis().set_label_coords(-0.05, 0.5)
                ax4.set_xlim((0, mask.sum()))
                ax4.set_yticks([0, 0.5, 1])
                ax4.set_yticklabels(['0', '0.5', '1'])
                ax4.grid(b=False, axis='x')

                ax4.set_xlabel("Genome position")
                x = [0] + list(chr_boundaries)
                ax4.set_xticks([0.5 * (x[i] + x[i + 1]) for i in range(len(x) - 1)])
                ax4.set_xticklabels(chromosomes, size=6)

                #plt.tight_layout()

                pdf.savefig()
                plt.close()

        # Plots the input channel
        with PdfPages(os.path.join(args.output, 'input_channels.pdf')) as pdf:
            for i in range(n_cells):

                plt.figure(figsize=(9, 5))

                for j in range(n_features):
                    ax = plt.subplot(100*n_outputs + 10 + j + 1)
                    ax.plot(X[i, mask, j].reshape((-1)), drawstyle='steps-mid')
                    ax.set_xlim((0, mask.sum()))
                    ax.vlines(chr_boundaries, 0, 1, linestyle='dashed', color='gray')
                    ax.tick_params(which='both', bottom='off', left='off', labelbottom='off', labelleft='off')
                pdf.savefig()
                plt.close()





















