import argparse

parser = argparse.ArgumentParser(description='myModel')
parser.add_argument('--feature-size', type=int, default=2048, help='size of feature (default: 2048)')
parser.add_argument('--batch-size', type=int, default=16, help='number of instances in a batch of data')
parser.add_argument('-d','--dataset', default='shanghai', help='dataset') # shanghai and ucf
parser.add_argument('-p', '--data_root', help='the root of testing data')
parser.add_argument('-c', '--checkpoint', help='the path  of the checkpoint')

