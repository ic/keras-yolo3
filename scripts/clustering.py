"""
Prepare custom anchors on you dataset

    python training.py \
        --path_dataset ../model_data/VOC_2007_train.txt \
        --path_output ../model_data

"""

import os
import sys
import logging
import argparse

sys.path += [os.path.abspath('.'), os.path.abspath('..')]
from keras_yolo3.kmeans import YOLO_Kmeans
from keras_yolo3.utils import check_params_path


def parse_params():
    # class YOLO defines the default value, so suppress any default HERE
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument('-d', '--path_dataset', type=str, required=True,
                        help='path to the train source - dataset,'
                             ' with single training instance per line')
    parser.add_argument('-o', '--path_output', type=str, required=True,
                        help='path to the output folder')
    parser.add_argument('--nb_clusters', type=int, required=False, default=9,
                        help='number of clusters')
    arg_params = vars(parser.parse_args())
    arg_params = check_params_path(arg_params)
    logging.debug('PARAMETERS: \n %s', repr(arg_params))
    return arg_params


def _main(path_dataset, path_output, nb_clusters=9):
    kmeans = YOLO_Kmeans(cluster_number=nb_clusters, filename=path_dataset)
    kmeans.txt2clusters(path_output)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    arg_params = parse_params()
    _main(**arg_params)
    logging.info('Done')
