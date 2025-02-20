from EPREL import *
import time
import pre_functions
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--craw', default='F')
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

    pre_functions.pre_process(df, args.method)