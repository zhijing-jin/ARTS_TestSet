# -*- coding:utf-8 -*-
# author: Xiaoyu Xing & Zhijing Jin
# datetime: 2020/6/3

import os
import argparse
from strategies import revTgt,revNon,addDiff


def get_args():
    parser = argparse.ArgumentParser('Settings for ARTS Generation')
    parser.add_argument('-dataset_name', default='laptop',
                        choices=['laptop', 'rest', 'mams'],
                        help='Choose a dataset from: laptop, mams, rest (which means the restaurant dataset)')
    parser.add_argument('-strategy', default='revTgt',
                        choices=['revTgt', 'revNon', 'addDiff'],
                        help='Choose a strategy from: RevTgt, RevNon, addDiff')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()

    dataset = args.dataset_name
    strategy = args.strategy

    data_folder='data/src_data/{}/'.format(dataset)
    input_file = os.path.join(data_folder, '2_test_sent.json')
    output_file = os.path.join(data_folder, 'test_adv.json')

    if strategy == 'revTgt':
        revTgt(data_folder,input_file, output_file)
    elif strategy == 'revNon':
        revNon(data_folder,input_file, output_file)
    elif strategy == 'addDiff':
        addDiff(dataset, os.path.join(data_folder, '2_train_sent.json'),
                os.path.join(data_folder, 'test_sent.json'), output_file)
