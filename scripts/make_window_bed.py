#!/usr/bin/env python

import argparse
import math

parser = argparse.ArgumentParser(description='Read in a fasts dict and produce a BED file describing equal sized bins on each chromosome')
parser.add_argument('--window', type=int,
                   help='The window size')
parser.add_argument('--dict',type=str,
                   help='Referecene genome dict file')

args = parser.parse_args()


with open(args.dict, 'r') as fin:
    for line in fin:
        if line[:3] == '@HD':
            continue
        _,sn,ln,*_ = line.split("\t")
        name, length = sn[3:], int(ln[3:])
        
        for i in range(math.ceil(length/args.window)-1):
            start, end = str(i*args.window), str((i+1)*args.window)
            print("\t".join([name, start, end, 
                             '{0}-{1}:{2}'.format(name,start, end)]))
                             
        start, end = str((i+1)*args.window), str(length)
        print("\t".join([name, start, end, 
                         '{0}-{1}:{2}'.format(name,start, end)]))
                             