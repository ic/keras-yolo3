"""
Evaluation of predictions againsts given dataset (in TXT format the same as training).
We expect that the predictions are in single folder and image names in dataset are the same

    python evaluate.py \
        --path_dataset ../model_data/VOC_2007_train.txt \
        --path_results ../results \
        --confidence 0.5 \
        --iou 0.5 \
        --visual

It generates
* statistic per image (mean over all classes)
* statistic per class (mean over all images)

See:
- https://github.com/rafaelpadilla/Object-Detection-Metrics
"""

import os
import sys
import argparse
import logging
from functools import partial
from pathos.multiprocessing import ProcessPool

import tqdm
import pandas as pd

sys.path += [os.path.abspath('.'), os.path.abspath('..')]
from keras_yolo3.utils import check_params_path, nb_workers, image_open, update_path
from keras_yolo3.model import compute_detect_metrics
from keras_yolo3.visual import draw_bounding_box

CSV_NAME_RESULTS_IMAGES = 'detection-results_conf=%.2f_iou=%.2f_stat-images.csv'
CSV_NAME_RESULTS_CLASSES = 'detection-results_conf=%.2f_iou=%.2f_stat-classes.csv'
ANNOT_COLUMNS = ('xmin', 'ymin', 'xmax', 'ymax', 'class')
# DETECT_COLUMNS = ('xmin', 'ymin', 'xmax', 'ymax', 'class', 'confidence')
TEMP_IMAGE_NAME = '%s_visual.jpg'


def parse_params():
    # class YOLO defines the default value, so suppress any default HERE
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument('-d', '--path_dataset', type=str, required=True,
                        help='path to the dataset, with single instance per line')
    parser.add_argument('-r', '--path_results', type=str, required=True,
                        help='path to the predictions')
    parser.add_argument('-c', '--confidence', type=float, required=False, default=0.5,
                        help='detection confidence score')
    parser.add_argument('--iou', type=float, required=False, default=0.5,
                        help='intersection over union')
    parser.add_argument('--nb_jobs', type=float, help='number of parallel processes',
                        default=0.9, required=False)
    parser.add_argument('--visual', default=False, action='store_true',
                        help='visualize annot & predict')
    arg_params = vars(parser.parse_args())
    arg_params = check_params_path(arg_params)
    logging.debug('PARAMETERS: \n %s', repr(arg_params))
    return arg_params


def draw_export_bboxes(img_path, path_out, bboxes_anot, bboxes_pred):
    img_name, _ = os.path.splitext(os.path.basename(img_path))
    image = image_open(img_path)

    for bb in bboxes_anot:
        image = draw_bounding_box(image, bb[4], bb[:4], swap_xy=True,
                                  color=(0, 255, 0), thickness=2)
    for bb in bboxes_pred:
        image = draw_bounding_box(image, bb[4], bb[:4], swap_xy=True,
                                  color=(255, 0, 0), thickness=2)

    name_visu = TEMP_IMAGE_NAME % img_name
    path_visu = os.path.join(update_path(path_out), name_visu)
    image.save(path_visu)
    return path_visu


def eval_image(line, path_results, thr_confidence=0.5, thr_iou=0.5, path_out=None):
    line_elems = line.strip().split()
    img_path = line_elems[0]
    img_name, _ = os.path.splitext(os.path.basename(img_path))

    path_pred = os.path.join(path_results, '%s.csv' % img_name)
    if not os.path.isfile(path_pred):
        return None

    boxes = [list(map(int, el.split(','))) for el in line_elems[1:]]
    df_annot = pd.DataFrame(boxes, columns=list(ANNOT_COLUMNS))
    if df_annot.empty:
        df_annot = pd.DataFrame(columns=ANNOT_COLUMNS)
    df_preds = pd.read_csv(path_pred, index_col=None)
    if df_preds.empty:
        df_preds = pd.DataFrame(columns=ANNOT_COLUMNS)

    # old version uses `score` instead `confidence`
    if 'confidence' not in df_preds.columns and 'score' in df_preds.columns:
        cols = df_preds.columns.tolist()
        idx = cols.index('score')
        df_preds.columns = cols[:idx] + ['confidence'] + cols[idx + 1:]
    # if confidence/score is defined, filter detections
    if 'confidence' in df_preds.columns:
        df_preds = df_preds[df_preds['confidence'] >= thr_confidence]
    # if class detection is not defined, assume everything as 0
    if 'class' not in df_preds.columns:
        df_preds['class'] = 0
    # in case one of DF does not have required columns skip it...
    all_cols_in_annot = all([c in df_annot.columns for c in ANNOT_COLUMNS])
    all_cols_in_pred = all([c in df_preds.columns for c in ANNOT_COLUMNS])
    if not (all_cols_in_annot and all_cols_in_pred):
        return None

    stats = compute_detect_metrics(df_annot[list(ANNOT_COLUMNS)], df_preds[list(ANNOT_COLUMNS)],
                                   iou_thresh=thr_iou)

    if path_out and os.path.isdir(path_out):
        draw_export_bboxes(img_path, path_out,
                           df_annot[list(ANNOT_COLUMNS)].values,
                           df_preds[list(ANNOT_COLUMNS)].values)

    # stats['name'] = img_name
    return stats


def _main(path_dataset, path_results, confidence, iou, visual=False, nb_jobs=0.9):
    with open(path_dataset, 'r') as fp:
        dataset = fp.readlines()

    if not dataset:
        logging.warning('Dataset is empty - %s', path_dataset)
        return

    nb_jobs = nb_workers(nb_jobs)
    pool = ProcessPool(nb_jobs) if nb_jobs > 1 else None
    _wrap_eval = partial(eval_image, path_results=path_results,
                         thr_confidence=confidence, thr_iou=iou,
                         path_out=path_results if visual else None)
    # multiprocessing loading of batch data
    map_process = pool.imap if pool else map

    results_image, results_class = [], []
    for stat in tqdm.tqdm(map_process(_wrap_eval, dataset), desc='Evaluation'):
        if not stat:
            continue
        results_image.append(dict(pd.DataFrame(stat).mean()))
        results_class += stat

    df_results_image = pd.DataFrame(results_image)
    logging.info(df_results_image.describe())
    path_csv = os.path.join(path_results, CSV_NAME_RESULTS_IMAGES % (confidence, iou))
    logging.debug('exporting csv: %s', path_csv)
    df_results_image.to_csv(path_csv)

    df_results_class = pd.DataFrame()
    for gr, df_gr in pd.DataFrame(results_class).groupby('class'):
        df_results_class = df_results_class.append(df_gr.mean(), ignore_index=True)
    logging.info(df_results_class)
    path_csv = os.path.join(path_results, CSV_NAME_RESULTS_CLASSES % (confidence, iou))
    logging.debug('exporting csv: %s', path_csv)
    df_results_class.to_csv(path_csv)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    pd.set_option("display.max_columns", 25)
    arg_params = parse_params()
    _main(**arg_params)
    logging.info('Done')
