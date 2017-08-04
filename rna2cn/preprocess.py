import matplotlib
matplotlib.use('Agg')

from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, MinMaxScaler
import os,re, glob
import pickle
import pandas as pd
import re,pyensembl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.decomposition
import argparse
import Bio.SeqIO, Bio.SeqUtils
import joblib
import warnings
import functools

warnings.filterwarnings('ignore', category=DeprecationWarning)

sns.set_style('whitegrid')

chromosomes = list(map(str, range(1,23)))# + ['X']

def fine_bin(data, ws, overlap_reduce, seq_dict, chromosomes=chromosomes):
    '''Takes an matrix input with four columns:

      chromosome, start, end, value
     
     for a list of region and duplicates the value 
     eg TPM or CN into ws sized bins accross each chromosome
     '''
    chr_lefts = {k: np.arange(ws, v, ws) for k, v in seq_dict.items()}
    chr_bins = {k: np.zeros((int(np.ceil(v/ws)))) for k, v in seq_dict.items()}
    chr_bins_count = {k: np.zeros((int(np.ceil(v/ws)))) for k, v in seq_dict.items()}

    for line in data:
        chr, s, e, x = line
        chr = str(int(chr))
        s_i, e_i = np.digitize([s,e], chr_lefts[chr])
        chr_bins[chr][s_i:e_i+1] = overlap_reduce(chr_bins[chr][s_i:e_i+1], chr_bins_count[chr][s_i:e_i+1], x)
        chr_bins_count[chr][s_i:e_i+1] += 1
    return [chr_bins[k] for k in chromosomes]

def getargs(argv):
    parser = argparse.ArgumentParser(prog='RNA2CN preprocess')
    parser.add_argument('--scRNA', required=True,
                        help='Input scRNA matrix')
    parser.add_argument('--normal', required=True, 
                        help='Input normal RNA matrix')
    parser.add_argument('--reference', required=True, 
                        help='Reference genome fasta')
    parser.add_argument('--window', required=True, type=int,
                        help='Genomic window size')
    parser.add_argument('--truncation', required=True, type=int,
                        help='The number of windows used for the a single TBPTT pass')
    parser.add_argument('--output', required=True)
    
    parser.add_argument('--tumour', required=False, default=None, 
                        help='Input tumour RNA matrix')
    parser.add_argument('--normalisation', 
                        choices=['none', 'independent', 'joint'],
                        default='none')
    parser.add_argument('--copy-number', required=False,
                        help='Tab separated file listing sample name and its matching target copy number profile for training',
                        default=None)
    parser.add_argument('--njobs', required=False, type=int, 
                        help='Number of parallel jobs to use to compute GC content', default=1)
    parser.add_argument('--gc', action='store_true', 
                       help='Include GC content feature')
                       
    #group = parser.add_mutually_exclusive_group()
    #group.add_argument('--tpm', action='store_true')
    #group.add_argument('--counts', action='store_true')
    
    return parser.parse_args(argv)

def get_gene_loci(table, dropchr='GKMXY'):
    '''Take a table indexed by geneid and add
       columns with the start,end,chr of each 
       gene. Drop any genes from chromosomes
       matching dropchr'''
    ensembl = pyensembl.ensembl_grch38
    genes = [ensembl.gene_by_id(x) for x in table.index]

    table['chr'] = [x.contig if x.contig[0] in dropchr else int(x.contig) for x in genes]
    table['start'] = [x.start for x in genes]
    table['end'] = [x.end for x in genes]

    table.drop( table[ table['chr'].apply(lambda x:str(x)[0] in dropchr) ].index, inplace=True)
    table = table.sort_values(['chr', 'start'])
    return table

def log1p_mean(x, n, y):
    '''Accumulate x with y where x has been 
       currently accumulated from n elements. x and y
       have been log(x+1) transformed so handle taking 
       their mean'''
    return np.log2(1 + (n*(2**x-1) + 2**y-1)/(n+1))
    
def load_expression(path, window, accumulate, seq_dict):
    '''Load the table of expression data per gene
       and bin it'''
    table = get_gene_loci(pd.read_csv(path, index_col=0))
    samples = [x for x in table.columns if x not in ['start', 'end', 'chr']] 
    binned = [fine_bin(table[['chr', 'start', 'end', k]].as_matrix(),
                          window,
                          accumulate, 
                          seq_dict) for k in samples]
    return binned, samples

def chr_gc(chr, reference, seq_dict, window):
    genome = Bio.SeqIO.index(reference, 'fasta')
    chr_bins = {k: list(zip(np.arange(0, v, window, dtype=int), np.concatenate([np.arange(window, v, window, dtype=int), [v+1]]))) for k, v in seq_dict.items()}
    gc = np.zeros(len(chr_bins[chr]))
    for i,(s,e) in enumerate(chr_bins[chr]):
        gc[i] = Bio.SeqUtils.GC(genome[chr][s:e].seq)
    return gc

def load_gc(reference, window, seq_dict, n_jobs):
    # Curry the gc content fuction with the reference path + window
    f = functools.partial(chr_gc, reference=reference, seq_dict=seq_dict, window=window)
    gc_bins = joblib.Parallel(n_jobs=n_jobs, verbose=1)(joblib.delayed(f)(chr) for chr in chromosomes)
    
    X_gc = np.array(pad_sequences(gc_bins, value=-1, padding='post', dtype='float'))
    X_gc[X_gc>0] = MinMaxScaler().fit_transform(X_gc[X_gc>0])
    X_gc = np.tile(X_gc, (1,26,1)).swapaxes(1,2).swapaxes(0,2)
    
    return X_gc

def load_dict(reference, dropchr='MKGYX'):
    
    if os.path.exists(reference+'.dict'):
        dict_path =reference+'.dict'
    elif os.path.exists(reference.rsplit('.', 1)[0] + '.dict'):
        dict_path = reference.rsplit('.', 1)[0] + '.dict'
    else:
        raise Exception("Unable to locate fasta dictionary for reference")
        
    seq_dict = {}
    with open(dict_path, 'r') as f:
        for line in f:
            if line[:3] == '@HD':
                continue
            _,sn,ln,*_ = line.split("\t")
            if sn[3] in dropchr:
                continue
            seq_dict[sn[3:]] = int(ln[3:])
    return seq_dict

def load_CN(cn_path, window, seq_dict):
    profile = fine_bin(pd.read_table(cn_path, header=None).as_matrix(), window, lambda x,n,y: y, seq_dict) 
    
    # Temporary hack: at the centromeres the CN spikes so replace this with -1
    for i,_ in enumerate(profile):
        profile[i][profile[i] > 5] = -1
        
    return pad_sequences(profile, value=-1, padding='post', dtype='float')
    
def subseq(X, seq_len, ws, seq_dict):
    
    n_chr, data = len(chromosomes), []    
    for i in range(int(X.shape[0]/n_chr)):
        chrX = [ X[ i*n_chr:(i+1)*n_chr ][j, :int(np.ceil(seq_dict[c]/ws)), :] for j,c in enumerate(chromosomes) ]
        chrX_padded = [pad_sequences(np.array_split(s, np.ceil(s.shape[0]/seq_len), axis=0), maxlen=seq_len, value=-2.0, padding='post', dtype='float') for s in chrX]
        padded = np.concatenate(chrX_padded, axis=0)
        m = (padded != -2)
        padded[padded == -2] = -1
        data.append(padded)
        
    X1 = np.stack(data)    
    return X1, m[:,:,0], list(map(len, chrX_padded))
    
def preprocess_command(argv):
    args = getargs(argv)
    
    print("Loading chromosome lengths")
    # Chromosome lengths
    seq_dict = load_dict(args.reference)
    
    # Expression data
    print("Loading single cell expressions")
    sc_binned, samples = load_expression(args.scRNA, args.window, log1p_mean, seq_dict)
    X_sc = np.concatenate([pad_sequences(x, value=-1, padding='post', dtype='float') for x in sc_binned], axis=0)
    n_samples = len(samples)
    
    print("Loading normal reference")
    normal_binned, _ = load_expression(args.normal, args.window, log1p_mean, seq_dict)
    X_normal = np.array([pad_sequences(x, value=-1, padding='post', dtype='float') for x in normal_binned]).mean(axis=0)
    X_normal = np.tile(X_normal, (n_samples,1))

    if args.tumour is not None:
        print("Loading bulk tumour")
        tumour_binned, _ = load_expression(args.tumour, args.window, log1p_mean, seq_dict)
        X_tumour = np.array([pad_sequences(x, value=-1, padding='post', dtype='float') for x in tumour_binned]).mean(axis=0)
        X_tumour = np.tile(X_tumour, (n_samples,1))
        X_expr = np.stack([X_sc, X_normal, X_tumour], axis=-1)
    else:
        X_expr = np.stack([X_sc, X_normal], axis=-1)

    # No normalisation is a drop through default
    if args.normalisation == 'joint':
        for i in range(X_expr.shape[0]):
            X_expr[i][X_expr[i] > 0] = RobustScaler(with_centering=False).fit_transform(X_expr[i][X_expr[i] > 0])
    
    elif args.normalisation == 'independent':
        for i in range(X_expr.shape[2]):
            for j in range(X_expr.shape[0]):
                X_expr[j,:,i][X_expr[j,:,i] > 0] = RobustScaler(with_centering=False).fit_transform(X_expr[j,:,i][X_expr[j,:,i] > 0])     
    
    X = X_expr
    
    # Start building up the feature matrix
    
    # GC content
    if args.gc:
        print("Loading GC content")
        X_gc = load_gc(args.reference, args.window, seq_dict, args.njobs)
        X = np.concatenate([X, X_gc], axis=-1)
    
    # End building up the feature matrix
    
    # Divide up and pad chromosomes
    X, X_mask, chr_breaks = subseq(X, args.truncation, args.window, seq_dict)
    
    # Target CNV profiles
    if args.copy_number is not None:
        print("Loading target CN profiles")
        with open(args.copy_number, 'r') as f:
            files = {x.split('\t')[0]: x.split("\t")[1].rstrip() for x in f}        
        Y = np.concatenate([load_CN(files[k], args.window, seq_dict) 
                            for k in samples], axis=0)
        Y = to_categorical(Y, 6).reshape((*Y.shape, 6))
        Y, Y_mask, _ = subseq(Y, args.truncation, args.window, seq_dict)

    else:
        Y, Y_mask = None, None
        
    # Dump output
    print("Writing output to " + args.output)
    with open(args.output, 'wb') as f:
        pickle.dump([X, X_mask, Y, Y_mask, chr_breaks, np.cumsum([np.ceil(seq_dict[c]/args.window) for c in chromosomes])], f)
