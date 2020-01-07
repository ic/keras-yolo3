"""
Clustering of bounding boxes
"""

import os
import logging

import numpy as np
import pandas as pd
import tqdm


class YOLO_Kmeans:
    """Clustering of boxes width and height

    Simplified version without file I/O

    >>> np.random.seed(0)
    >>> boxes = np.random.randint(50, 250, (100, 2))
    >>> clust = YOLO_Kmeans(cluster_number=3, filename=None)
    >>> result = clust.fit(boxes, k=clust.cluster_number)
    >>> result = result[np.lexsort(result.T[0, None])]
    >>> result
    array([[ 99, 192],
           [147,  81],
           [198, 179]])
    >>> clust.avg_iou(boxes, result)  # doctest: +ELLIPSIS
    0.688...
    """

    def __init__(self, cluster_number, filename):
        self.cluster_number = cluster_number
        self.filename = filename

    def iou(self, boxes, clusters):  # 1 box -> k clusters
        n = boxes.shape[0]
        k = self.cluster_number

        box_area = boxes[:, 0] * boxes[:, 1]
        box_area = box_area.repeat(k)
        box_area = np.reshape(box_area, (n, k))

        cluster_area = clusters[:, 0] * clusters[:, 1]
        cluster_area = np.tile(cluster_area, [1, n])
        cluster_area = np.reshape(cluster_area, (n, k))

        box_w_matrix = np.reshape(boxes[:, 0].repeat(k), (n, k))
        cluster_w_matrix = np.reshape(np.tile(clusters[:, 0], (1, n)), (n, k))
        min_w_matrix = np.minimum(cluster_w_matrix, box_w_matrix)

        box_h_matrix = np.reshape(boxes[:, 1].repeat(k), (n, k))
        cluster_h_matrix = np.reshape(np.tile(clusters[:, 1], (1, n)), (n, k))
        min_h_matrix = np.minimum(cluster_h_matrix, box_h_matrix)
        inter_area = np.multiply(min_w_matrix, min_h_matrix)

        result = inter_area / (box_area + cluster_area - inter_area)
        return result

    def avg_iou(self, boxes, clusters):
        accuracy = np.mean([np.max(self.iou(boxes, clusters), axis=1)])
        return accuracy

    def fit(self, boxes, k, dist=np.median):
        box_number = boxes.shape[0]
        # distances = np.empty((box_number, k))
        last_nearest = np.zeros((box_number,))
        clusters = boxes[np.random.choice(box_number, k, replace=False)]  # init k clusters

        while True:
            distances = 1 - self.iou(boxes, clusters)
            current_nearest = np.argmin(distances, axis=1)
            if (last_nearest == current_nearest).all():
                break  # clusters won't change
            for cluster in range(k):
                clusters[cluster] = dist(  # update clusters
                    boxes[current_nearest == cluster], axis=0)
            last_nearest = current_nearest

        return clusters

    def result2csv(self, data, path_out):
        name, _ = os.path.splitext(os.path.basename(self.filename))
        path_csv = os.path.join(path_out, name + '_anchors.csv')
        logging.debug('Export CSV: %s', path_csv)
        pd.DataFrame(data, dtype=int).to_csv(path_csv, header=None, index=None)

    def txt2boxes(self):
        with open(self.filename, 'r') as fp:
            lines = fp.readlines()
        dataset = []
        for line in tqdm.tqdm(lines, desc='Extract bboxes'):
            infos = line.strip().split(' ')
            for bbox in infos[1:]:
                box = list(map(float, bbox.split(',')))
                width = abs(box[2] - box[0])
                height = abs(box[3] - box[1])
                dataset.append([width, height])
        bboxes_wh = np.array(dataset)
        return bboxes_wh

    def txt2clusters(self, path_out):
        all_boxes = self.txt2boxes()
        result = self.fit(all_boxes, k=self.cluster_number)
        result = result[np.lexsort(result.T[0, None])]
        self.result2csv(result, path_out)
        logging.info('K anchors:\n %s', repr(result))
        logging.info('Accuracy: %f', self.avg_iou(all_boxes, result) * 100)
        return result
