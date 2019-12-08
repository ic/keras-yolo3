"""
Detections script::

    python detection.py \
        --path_weights ./model_data/yolo3-tiny.h5 \
        --path_anchors ./model_data/tiny-yolo_anchors.csv \
        --path_classes ./model_data/coco_classes.txt \
        --path_output ./results \
        --path_image ./model_data/bike-car-dog.jpg \
        --path_video person.mp4

You can run detection on whole folder with images/videos::

    python detection.py \
        --path_weights ./model_data/yolo3-tiny.h5 \
        --path_anchors ./model_data/tiny-yolo_anchors.csv \
        --path_classes ./model_data/coco_classes.txt \
        --path_output ./results \
        --path_image ./model_data/*.jpg \
        --path_video /samples/*.mp4

"""

import os
import sys
import argparse
import logging
import json
import time
import glob

import cv2
import tqdm
from PIL import Image
import pandas as pd
import numpy as np

sys.path += [os.path.abspath('.'), os.path.abspath('..')]
from keras_yolo3.yolo import YOLO
from keras_yolo3.utils import update_path

VISUAL_EXT = '_detect'
VIDEO_FORMAT = cv2.VideoWriter_fourcc('F', 'M', 'P', '4')


def arg_params_yolo():
    # class YOLO defines the default value, so suppress any default HERE
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    # Command line options
    parser.add_argument('-a', '--path_anchors', type=str, required=True,
                        help='path to anchor definitions')
    parser.add_argument('-c', '--path_classes', type=str,
                        help='path to class definitions')
    parser.add_argument('--nb_gpu', type=int, help='Number of GPU to use',
                        default=str(YOLO.get_defaults("nb_gpu")))
    parser.add_argument('-o', '--path_output', required=False, type=str, default='.',
                        help='path to the output directory')
    return parser


def parse_params():
    # class YOLO defines the default value, so suppress any default HERE
    parser = arg_params_yolo()
    parser.add_argument('-w', '--path_weights', type=str, required=True,
                        help='path to model weight file')
    parser.add_argument('-i', '--path_image', nargs='*', type=str, required=False,
                        help='Images to be processed (sequence of paths)')
    parser.add_argument('-v', '--path_video', nargs='*', type=str, required=False,
                        help='Video to be processed (sequence of paths)')
    arg_params = vars(parser.parse_args())
    for k_name in ('path_image', 'path_video'):
        # if there is only single path still make it as a list
        if k_name in arg_params and not isinstance(arg_params[k_name], (list, tuple)):
            arg_params[k_name] = [arg_params[k_name]]
    # Update paths
    for k in (k for k in arg_params if 'path' in k):
        if k in ('path_image', 'path_video'):
            arg_params[k] = [update_path(path_) for path_ in arg_params[k]]
        elif arg_params[k]:
            arg_params[k] = update_path(arg_params[k])
            assert os.path.exists(arg_params[k]), 'missing (%s): %s' % (k, arg_params[k])
    logging.debug('PARAMETERS: \n %s', repr(arg_params))
    return arg_params


def predict_image(yolo, path_image, path_output=None):
    path_image = update_path(path_image)
    if not path_image:
        logging.debug('no image given')
    elif not os.path.isfile(path_image):
        logging.warning('missing image: %s', path_image)

    image = Image.open(path_image)
    image_pred, pred_items = yolo.detect_image(image)
    if path_output is None or not os.path.isdir(path_output):
        image_pred.show()
    else:
        name = os.path.splitext(os.path.basename(path_image))[0]
        path_out_img = os.path.join(path_output, name + VISUAL_EXT + '.jpg')
        path_out_csv = os.path.join(path_output, name + '.csv')
        logging.debug('exporting image: "%s" and detection: "%s"',
                      path_out_img, path_out_csv)
        image_pred.save(path_out_img)
        pd.DataFrame(pred_items).to_csv(path_out_csv)


def predict_video(yolo, path_video, path_output=None, show_stream=False):
    try:
        path_video = int(path_video)
    except Exception:  # not using web cam
        path_video = update_path(path_video)
    else:  # using the (infinite) stream add option to terminate
        show_stream = True

    # Create a video capture object to read videos
    try:
        video = cv2.VideoCapture(path_video)
    except Exception:
        logging.warning('missing: %s', path_video)
        return

    if path_output is not None and os.path.isdir(path_output):
        video_fps = video.get(cv2.CAP_PROP_FPS)
        video_size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
                      int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        name = os.path.splitext(os.path.basename(path_video))[0] \
            if isinstance(path_video, str) else str(path_video)
        path_out = os.path.join(path_output, name + VISUAL_EXT + '.avi')
        logging.debug('export video: %s', path_out)
        out_vid = cv2.VideoWriter(path_out, VIDEO_FORMAT, video_fps, video_size)
        frame_preds = []
        path_json = os.path.join(path_output, name + '.json')
    else:
        out_vid, frame_preds, path_json = None, None, None

    while video.isOpened():
        success, frame = video.read()
        if not success:
            logging.warning('video read status: %r', success)
            break
        image = Image.fromarray(frame)
        t_start = time.time()
        image_pred, pred_items = yolo.detect_image(image)
        frame = np.asarray(image_pred)
        fps = 'FPS: %f' % (1. / (time.time() - t_start))
        cv2.putText(frame, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)

        if out_vid:
            out_vid.write(frame)
            frame_preds.append(pred_items)
            with open(path_json, 'w') as fp:
                json.dump(frame_preds, fp)
        if show_stream:
            cv2.imshow('YOLOv3', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyWindow('YOLOv3')
                break

    if out_vid:
        out_vid.release()
        logging.debug('exported predictions: %s', path_json)


def expand_file_paths(paths):
    paths_unrolled = []
    for ph in paths:
        if '*' in ph:
            paths_unrolled += glob.glob(ph)
        elif os.path.isfile(ph):
            paths_unrolled.append(ph)
    return paths_unrolled


def _main(path_weights, path_anchors, path_classes, path_output, nb_gpu=0, **kwargs):

    yolo = YOLO(weights_path=path_weights, anchors_path=path_anchors,
                classes_path=path_classes, nb_gpu=nb_gpu)

    logging.info('Start image/video processing..')
    if 'path_image' in kwargs:
        paths_img = expand_file_paths(kwargs['path_image'])
        for path_img in tqdm.tqdm(paths_img, desc='images'):
            logging.debug('processing: "%s"', path_img)
            predict_image(yolo, path_img, path_output)
    if 'path_video' in kwargs:
        paths_vid = expand_file_paths(kwargs['path_video'])
        for path_vid in tqdm.tqdm(paths_vid, desc='videos'):
            logging.debug('processing: "%s"', path_vid)
            predict_video(yolo, path_vid, path_output)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    # class YOLO defines the default value, so suppress any default HERE
    arg_params = parse_params()

    _main(**arg_params)

    logging.info('Done')
