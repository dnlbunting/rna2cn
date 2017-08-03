import sys
import pickle
import argparse

def main():
    main_parser = argparse.ArgumentParser(prog='RNA2CN')
    main_parser.add_argument('command', choices=['train', 'preprocess', 'predict', 'evaluate'])
                                        
    main_args = main_parser.parse_args(sys.argv[1:2])
    if main_args.command == 'train':
        import rna2cn.training
        rna2cn.training.train_command(sys.argv[2:])
    else:
        print("Yeah I haven't written this yet")
        sys.exit(1)
        
if __name__ == '__main__':
    main()

        
