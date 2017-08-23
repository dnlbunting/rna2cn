import sys
import pickle
import argparse

import os
os.environ['KERAS_BACKEND'] = 'theano'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['THEANO_FLAGS'] = "openmp=False"


def main():
    main_parser = argparse.ArgumentParser(prog='RNA2CN',
                                          formatter_class=argparse.RawTextHelpFormatter,
                                          description='''RNA2CN toolkit for predicting copy number aberrations from scRNAseq.

Subcommands
------------

preprocess - Encodes input expression data and optionally target CNA profiles into features for model training/evaluation
concatenate - Combine together multiple datasets separately prepared using the preprocess command
train - Trains a LSTM neural network defined in Keras JSON format using the prepared input data
evaluate - Evaluates the performance of the trained network on the test data and produces output graphs.

''')
    main_parser.add_argument('command', choices=['train', 'preprocess', 'predict', 'evaluate', 'concatenate'])

    main_args = main_parser.parse_args(sys.argv[1:2])

    if main_args.command == 'train':
        import rna2cn.training
        rna2cn.training.train_command(sys.argv[2:])

    elif main_args.command == 'preprocess':
        import rna2cn.preprocess
        rna2cn.preprocess.preprocess_command(sys.argv[2:])

    elif main_args.command == 'evaluate':
        import rna2cn.evaluate
        rna2cn.evaluate.evaluate_command(sys.argv[2:])

    elif main_args.command == 'concatenate':
        import rna2cn.concatenate
        rna2cn.concatenate.concatenate_command(sys.argv[2:])

    else:
        print("Yeah I haven't written this yet")
        sys.exit(1)

if __name__ == '__main__':
    main()
