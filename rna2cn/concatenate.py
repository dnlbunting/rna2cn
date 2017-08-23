import pickle
import argparse
import numpy as np


def getargs(argv):
    parser = argparse.ArgumentParser(prog='RNA2CN concatenate')

    parser.add_argument('--output', required=True)
    parser.add_argument('samples', nargs=2)

    return parser.parse_args(argv)


def args_match(args1, args2):
    match = ['gc', 'normalisation', 'reference', 'truncation', 'window']
    return all([args1.__getattribute__(x) == args2.__getattribute__(x) for x in match])


def concatenate_command(argv):
    concat_args = getargs(argv)
    for i, data in enumerate(concat_args.samples):
        print("Loading " + data)
        with open(data, 'rb') as f:
            X_, Y_, mask_, train_cells_, chr_breaks_, chr_boundaries_, args_ = pickle.load(f)
            if i == 0:
                (X, Y, mask, train_cells, chr_breaks,
                 chr_boundaries, args) = (X_, Y_, mask_, train_cells_,
                                          chr_breaks_, chr_boundaries_, args_)
                continue

            train_cells = np.concatenate((train_cells, train_cells_ + Y.shape[0]), axis=0)  # offset the cell indices
            X = np.concatenate((X, X_), axis=0)
            Y = np.concatenate((Y, Y_), axis=0)

            mask_match = mask == mask_
            chr_breaks_match = chr_breaks_ == chr_breaks
            chr_boundaries_match = chr_boundaries_ == chr_boundaries

            if not np.all([mask_match.all(), chr_breaks_match, chr_boundaries_match.all(), args_match(args, args_)]):
                raise Exception("All parameters of the datasets must match!!")

    print("Writing to " + concat_args.output)
    with open(concat_args.output, 'wb') as f:
        pickle.dump([X, Y, mask_, train_cells, chr_breaks_, chr_boundaries_, args_], f)
