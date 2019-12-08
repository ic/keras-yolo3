"""
Creating training file from RectLabel

>> python annotation_rectlabel.py \
    --path_annot ~/Annotations \
    --root_images ~/Images \
    --path_output ~/result.txt
"""

import os
import sys
import logging
import argparse
import xml.etree.ElementTree as ET

import tqdm

sys.path += [os.path.abspath('.'), os.path.abspath('..')]
from keras_yolo3.utils import update_path


def parse_arguments():
    parser = argparse.ArgumentParser(description='Annotation Converter (RectLabel).')
    parser.add_argument('--root_annot', type=str, required=True,
                        help='Root path to annotation files.')
    parser.add_argument('--root_images', type=str, required=True,
                        help='Root folder to images of dataset.')
    parser.add_argument('--output', type=str, required=False, default='.',
                        help='Output file.')
    arg_params = vars(parser.parse_args())
    for k in (k for k in arg_params if 'path' in k):
        arg_params[k] = update_path(arg_params[k])
        assert os.path.exists(arg_params[k]), 'missing (%s): %s' % (k, arg_params[k])
    logging.info('PARAMETERS: \n%s', '\n'.join(['"%s": \t\t %r' % (k, arg_params[k])
                                                for k in arg_params]))
    return arg_params


def _main(root_annot, root_images, output):
    logging.info('loading annotations "%s"', root_annot)
    for path_annot in tqdm.tqdm([os.path.join(root_annot, f) for f in os.listdir(root_annot) if f.endswith('xml')]):
        _process(path_annot, root_images, output)
    logging.info('Done.')

def _process(path_annot, root_images, output):
    tree = ET.parse(path_annot)
    root = tree.getroot()
    filename = os.path.join(root_images, root.find('filename').text)
    with open(output, 'a') as out:
        out.write(filename)
        for box in root.findall('object'):
            out.write(' ')
            bb = box.find('bndbox')
            out.write(','.join([
                bb.find('xmin').text,
                bb.find('ymin').text,
                bb.find('xmax').text,
                bb.find('ymax').text,
                '0']))
        out.write('\n')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    arg_params = parse_arguments()
    _main(**arg_params)
