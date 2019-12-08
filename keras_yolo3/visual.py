"""
Visualisation module

"""

import logging

import numpy as np
from PIL import ImageDraw
import matplotlib.pylab as plt


def draw_bounding_box(image, predicted_class, box, score=None, color=(255, 0, 0), thickness=1, swap_xy=False):
    """draw bounding box

    :param ndarray image:
    :param int|str predicted_class:
    :param list box:
    :param float score:
    :param tuple(int,int,int) color:
    :param int thickness:
    :return ndarray:

    >>> import os
    >>> from keras_yolo3.utils import update_path, image_open
    >>> img = image_open(os.path.join(update_path('model_data'), 'bike-car-dog.jpg'))
    >>> draw = draw_bounding_box(img, 1, [150, 200, 250, 300], 0.9, color=(255, 0, 0), thickness=3)
    >>> draw  # doctest: +ELLIPSIS
    <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=520x518 at ...>
    """
    label_score = '{} ({:.2f})'.format(predicted_class, score) if score else ''

    draw = ImageDraw.Draw(image)
    log_level = logging.getLogger().getEffectiveLevel()
    logging.getLogger().setLevel(logging.INFO)
    label_size = draw.textsize(label_score)
    logging.getLogger().setLevel(log_level)

    if swap_xy:
        box = np.asarray(box)[[1, 0, 3, 2]]

    top, left, bottom, right = box
    top = max(0, np.floor(top + 0.5).astype('int32'))
    left = max(0, np.floor(left + 0.5).astype('int32'))
    bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
    right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
    logging.debug(' >> drawing bbox > %s: (%i, %i), (%i, %i)', label_score, left, top, right, bottom)

    if top - label_size[1] >= 0:
        text_origin = np.array([left, top - label_size[1]])
    else:
        text_origin = np.array([left, top + 1])

    for i in range(thickness):
        draw.rectangle([left + i, top + i, right - i, bottom - i],
                       outline=color)
    draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)],
                   fill=color)
    draw.text(list(text_origin), label_score, fill=(0, 0, 0))
    del draw
    return image


def _draw_bbox(ax, bbox, color='r'):
    x_min, y_min, x_max, y_max = bbox[:4]
    if (x_max - x_min) * (y_max - y_min) == 0:
        return
    x = [x_min, x_min, x_max, x_max, x_min]
    y = [y_min, y_max, y_max, y_min, y_min]
    ax.plot(x, y, color=color)


def show_augment_data(image_in, bboxes, img_data, box_data, title=''):
    """visualise image_data and box_data

    :param ndaaray image_in: original image
    :param ndaaray bboxes: original annotation
    :param ndaaray img_data: augmented image
    :param ndaaray box_data: adjusted boxes
    :return:

    >>> img = np.random.random((250, 200, 3))
    >>> box = [10, 40, 50, 90, 0]
    >>> show_augment_data(img, [box], img, [box, [0] * 5])  # doctest: +ELLIPSIS
    <...>
    """
    fig, axarr = plt.subplots(ncols=2, figsize=(14, 8), constrained_layout=True)
    axarr[0].set_title('Original image with annotation')
    axarr[0].imshow(np.array(image_in))
    for box in bboxes:
        _draw_bbox(axarr[0], box)
    axarr[1].set_title('Augmented image with adjusted bounding boxes')
    axarr[1].imshow(np.array(img_data), vmin=0, vmax=1)
    for box in box_data:
        _draw_bbox(axarr[1], box)
    fig.suptitle(title)
    fig.tight_layout()
    return fig
