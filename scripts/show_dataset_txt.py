"""
Visualize the input dataset

    python show_dataset_txt.py \
        --path_dataset ../model_data/VOC_2007_train.txt \
        --path_output ../results

"""

import os
import sys
import argparse
import logging
from functools import partial
from pathos.multiprocessing import ProcessPool

import tqdm

sys.path += [os.path.abspath('.'), os.path.abspath('..')]
from keras_yolo3.utils import check_params_path, nb_workers, image_open, update_path
from keras_yolo3.visual import draw_bounding_box


ANNOT_COLUMNS = ('xmin', 'ymin', 'xmax', 'ymax', 'class')
TEMP_IMAGE_NAME = '%s_visual.jpg'


def parse_params():
    # class YOLO defines the default value, so suppress any default HERE
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument('-d', '--path_dataset', type=str, required=True,
                        help='path to the dataset, with single instance per line')
    parser.add_argument('-o', '--path_output', type=str, required=True,
                        help='path to folder for export')
    parser.add_argument('--nb_jobs', type=float, help='number of parallel processes',
                        default=0.9, required=False)
    arg_params = vars(parser.parse_args())
    arg_params = check_params_path(arg_params)
    logging.debug('PARAMETERS: \n %s', repr(arg_params))
    return arg_params


def parse_draw_export_bboxes(line, path_out):
    line_elems = line.strip().split()
    img_path = update_path(line_elems[0])
    img_name, _ = os.path.splitext(os.path.basename(img_path))

    image = image_open(img_path)
    name_visu = TEMP_IMAGE_NAME % img_name
    path_visu = os.path.join(update_path(path_out), name_visu)

    boxes = [list(map(int, el.split(','))) for el in line_elems[1:]]
    for bb in boxes:
        image = draw_bounding_box(image, bb[4], bb[:4], swap_xy=True,
                                  color=(0, 255, 0), thickness=2)

    # from keras_yolo3.visual import show_augment_data
    # show_augment_data(image, boxes, image, boxes)

    image.save(path_visu)
    return path_visu


def _main(path_dataset, path_output, nb_jobs=0.9):
    assert os.path.isfile(path_dataset)
    with open(path_dataset, 'r') as fp:
        dataset = fp.readlines()

    if not dataset:
        logging.warning('Dataset is empty - %s', path_dataset)
        return

    nb_jobs = nb_workers(nb_jobs)
    pool = ProcessPool(nb_jobs) if nb_jobs > 1 else None
    _wrap_visu = partial(parse_draw_export_bboxes, path_out=path_output)

    # multiprocessing loading of batch data
    map_process = pool.imap if pool else map
    _ = list(tqdm.tqdm(map_process(_wrap_visu, dataset), desc='Visualization'))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    arg_params = parse_params()
    _main(**arg_params)
    logging.info('Done')
