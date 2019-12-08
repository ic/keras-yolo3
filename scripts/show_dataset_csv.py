"""
Visualize the input dataset

    python show_dataset_csv.py \
        --pattern_images ../model_data/images/*.png \
        --path_csv_folder ../model_data/annot \
        --path_output ../results

"""

import os
import sys
import glob
import argparse
import logging
from functools import partial
from pathos.multiprocessing import ProcessPool

import tqdm
import pandas as pd

sys.path += [os.path.abspath('.'), os.path.abspath('..')]
from keras_yolo3.utils import check_params_path, nb_workers, image_open, update_path
from keras_yolo3.visual import draw_bounding_box

ANNOT_COLUMNS = ('xmin', 'ymin', 'xmax', 'ymax', 'class')
TEMP_IMAGE_NAME = '%s_visual.jpg'


def parse_params():
    # class YOLO defines the default value, so suppress any default HERE
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument('-i', '--pattern_images', type=str, required=True,
                        help='path to the folder with images')
    parser.add_argument('-b', '--path_csv_folder', type=str, required=True,
                        help='path to the folder with bounding boxes')
    parser.add_argument('-o', '--path_output', type=str, required=True,
                        help='path to folder for export')
    parser.add_argument('--nb_jobs', type=float, help='number of parallel processes',
                        default=0.9, required=False)
    arg_params = vars(parser.parse_args())
    arg_params = check_params_path(arg_params)
    logging.debug('PARAMETERS: \n %s', repr(arg_params))
    return arg_params


def load_draw_export_bboxes(img_path, path_bboxes, path_out):
    img_name, _ = os.path.splitext(os.path.basename(img_path))

    image = image_open(img_path)
    name_visu = '%s_visual.jpg' % img_name
    path_visu = os.path.join(update_path(path_out), name_visu)

    path_csv = os.path.join(path_bboxes, '%s.csv' % img_name)
    if not os.path.isfile(path_csv):
        return
    df_boxes = pd.read_csv(path_csv)
    if 'class' not in df_boxes.columns:
        df_boxes['class'] = 0
    for bb in df_boxes[list(ANNOT_COLUMNS)].values:
        image = draw_bounding_box(image, bb[4], bb[:4], swap_xy=True,
                                  color=(0, 255, 0), thickness=2)

    # from keras_yolo3.visual import show_augment_data
    # show_augment_data(image, boxes, image, boxes)

    image.save(path_visu)
    return path_visu


def _main(pattern_images, path_csv_folder, path_output, nb_jobs=0.9):
    pattern_images = os.path.join(update_path(os.path.dirname(pattern_images)),
                                  os.path.basename(pattern_images))
    assert os.path.isdir(os.path.dirname(pattern_images))
    image_paths = sorted(glob.glob(pattern_images))

    if not image_paths:
        logging.warning('Dataset is empty for %s', pattern_images)
        return

    nb_jobs = nb_workers(nb_jobs)
    pool = ProcessPool(nb_jobs) if nb_jobs > 1 else None
    _wrap_visu = partial(load_draw_export_bboxes, path_bboxes=path_csv_folder, path_out=path_output)

    # multiprocessing loading of batch data
    map_process = pool.imap if pool else map
    _ = list(tqdm.tqdm(map_process(_wrap_visu, image_paths), desc='Visualization'))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    arg_params = parse_params()
    _main(**arg_params)
    logging.info('Done')
