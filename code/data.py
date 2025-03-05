from EPREL import *
import time
import pre_functions
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--craw', default='all')
parser.add_argument('--method', default='filter')
args = parser.parse_args()

if args.craw == 'T':
    start_time = time.time()

    df = get_EPREL_DB_v3()
    df.T.to_csv('../data/data_original.csv', index=False)

    end_time = time.time()
    elapse_time = end_time - start_time
    print(f"Crawling Time : {elapse_time} sec")

elif args.craw == 'F':
    df = pd.read_csv('../data/data_original.csv')

    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    output_path = os.path.join(parent_dir, "output")
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    pre_functions.pre_process(df, args.method)

elif args.craw == 'all':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    output_path = os.path.join(parent_dir, "output")
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    df = get_EPREL_DB_v3()
    df.T.to_csv('../data/data_original.csv', index=False)

    pre_functions.pre_process(df.T, 'IF', "../output/pre_data_")
