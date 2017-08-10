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
                        default='accuracy,breakpoints,events')
    parser.add_argument('--confusion', action='store_true',
                        help='Plot the confusion matrix')
    parser.add_argument('--precision-recall', action='store_true',
                        help='Plot the precision recall curve')
    parser.add_argument('--plot', action='store_true',
                        help='Plot the predicted copy number profiles')

    return parser.parse_args(argv)


def predict(model, X, chr_steps, n_outputs):
    chroffset = [0] + list(np.cumsum(chr_steps))
    pred = np.zeros((*X.shape[:-1], n_outputs))
    for s in range(X.shape[0]):
        for i, chr in list(enumerate(chromosomes)):
            model.reset_states()
            for j in range(chr_steps[i]):
                x = X[s:s + 1, chroffset[i] + j]
                pred[s, chroffset[i] + j] = model.predict(x, verbose=0, batch_size=1)
    return pred


def get_breakpoints(y):
    events = []
    for i, x in enumerate(y):
        if i != 0 and y[i] != y[i - 1]:
            events.append(i)
    return np.array(events)


def get_singletons(y):
    singletons = []
    for i, x in enumerate(y):
        if i != 0 and y[i] != y[i - 1]:
            if i != len(y) - 1 and y[i + 1] != y[i]:
                singletons.append(i)
    return np.array(singletons)


def makedata(X, Y, train_cells):
    '''Split X,Y cellwise into train and test sets,
       returns (X_train, Y_train, X_test, Y_test, test_cells)'''
    test_cells = np.array([x for x in range(len(X)) if x not in train_cells])

    X_train, Y_train = X[train_cells], Y[train_cells]
    X_test, Y_test = X[test_cells], Y[test_cells]

    return X_train, Y_train, X_test, Y_test, test_cells


def evaluate_command(argv):
    args = getargs(argv)

    with open(args.model, 'r') as f:
        model = model_from_json(f.read())
    print("Loaded model from file " + args.model)
    print(model.summary())

    model.load_weights(args.weights)
    print("Loaded weights from " + args.weights)

    with open(args.data, 'rb') as f:
        X, Y_oh, mask, train_cells, chr_steps, chr_boundaries, _ = pickle.load(f)

    n_cells, n_points, n_outputs = Y_oh.shape[0], Y_oh.shape[2], Y_oh.shape[-1]
    n_features = X.shape[-1]
    Y = Y_oh.argmax(axis=-1)[:, mask].reshape((n_cells, -1))  # Y comes in one hot categorical encoding

    pred = predict(model, X, chr_steps, n_outputs)
    yhat = pred.argmax(axis=-1)[:, mask].reshape((n_cells, -1))

    # Probability of there being a CN != 2 event at this site
    event_Y = np.where(Y_oh[..., 2], 0, 1)[:, mask]
    event_p = 1 - pred[:, mask, 2].reshape((n_cells, -1))

    metrics = {k: None for k in args.metrics.split(',')}

    if 'accuracy' in metrics.keys():
        metrics['accuracy'] = sklearn.metrics.accuracy_score(Y.ravel(), yhat.ravel())
        metrics['mse'] = sklearn.metrics.mean_squared_error(Y.ravel(), yhat.ravel())

    if 'events' in metrics.keys():
        metrics['auc'] = sklearn.metrics.roc_auc_score(event_Y.ravel(), event_p.ravel())
        metrics['event_accuracy'] = sklearn.metrics.accuracy_score(event_Y.ravel(), (event_p > 0.5).ravel())

    if 'breakpoints' in metrics.keys():
        dist = int(0.1 * yhat.shape[1])
        metrics['n_breakpoints'] = np.array([len(get_breakpoints(y)) for y in yhat])
        # metrics['concordant_breakpoints'] = np.mean([np.abs(get_breakpoints(yh) - get_breakpoints(y)) < dist for yh, y in zip(yhat, Y)])
        metrics['singletons'] = np.array([len(get_singletons(y)) for y in yhat])
    print(metrics)

    if args.confusion:
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
        precision, recall, thresholds = sklearn.metrics.precision_recall_curve(event_Y.ravel(), event_p.ravel())
        plt.plot(precision, recall)
        plt.plot([0, 1], [1, 0], '--', color='gray')
        plt.xlabel("Precision")
        plt.ylabel("Recall")
        plt.title("Event detection")
        plt.savefig(os.path.join(args.output, 'prec_rec.pdf'))
        plt.close()

    if args.plot:
        with PdfPages(os.path.join(args.output, 'profiles.pdf')) as pdf:
            for i in range(n_cells):

                plt.figure(figsize=(9, 5))
                gridspec = (19, 1)

                ########## Subplot 1 #############
                ax1 = plt.subplot2grid(gridspec, (0, 0), rowspan=1)
                for j in range(n_outputs):
                    ax1.plot(Y_oh[i, mask, j].reshape((-1)), 's', linewidth=10)
                ax1.set_ylim((0.95, 1.05))
                ax1.set_xlim((0, mask.sum()))
                ax1.set_ylabel("Y", rotation=0, va='center')
                ax1.get_yaxis().set_label_coords(-0.05, 0.5)
                ax1.tick_params(which='both', bottom='off', left='off', labelbottom='off', labelleft='off')
                ax1.spines['top'].set_visible(False)
                ax1.spines['right'].set_visible(False)
                ax1.spines['left'].set_visible(False)
                ax1.spines['bottom'].set_visible(False)
                plt.title('Cell {0} - {1}'.format(i, 'Train' if i in train_cells else 'Test' ))
                ax1.grid()

                ########## Subplot 2 #############
                ax2 = plt.subplot2grid(gridspec, (1, 0), rowspan=6)
                for j in range(n_outputs):
                    ax2.plot(pred[i, mask, j].reshape((-1)), label=str(j))
                ax2.vlines(chr_boundaries, 0, 1, linestyle='dashed', color='gray')
                ax2.set_ylim((0, 1))
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
                ax3.plot(Y[i], '.', color='black', ms=1)
                #ax3.plot(yhat[i], '.', color='red', ms=1)
                ax3.set_ylabel("Copy number")
                ax3.get_yaxis().set_label_coords(-0.05, 0.5)
                ax3.vlines(chr_boundaries, 0, 5, linestyle='dashed', color='gray')
                ax3.set_ylim((0, n_outputs - 1))
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





















