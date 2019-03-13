# coding: utf-8
import numpy as np


def iou(boxes, clusters):   # 1 box -> k clusters
    boxes = np.expand_dims(boxes, -1)
    box_area = boxes[:, 0] * boxes[:, 1]
    cluster_area = clusters[:, 0] * clusters[:, 1]

    min_w_matrix = np.minimum(clusters[:, 0], boxes[:, 0])
    min_h_matrix = np.minimum(clusters[:, 1], boxes[:, 1])
    inter_area = np.multiply(min_w_matrix, min_h_matrix)

    result = inter_area / (box_area + cluster_area - inter_area)
    return result


def kmeans(boxes, k):
    n = boxes.shape[0]
    last_nearest = np.zeros((n,))
    clusters = boxes[np.random.choice(n, k, replace=False)]  # init k clusters
    while True:
        distances = 1 - iou(boxes, clusters)

        current_nearest = np.argmin(distances, axis=1)
        if (last_nearest == current_nearest).all():
            break  # clusters won't change
        for cluster in range(k):
            # update clusters
            clusters[cluster] = np.median(boxes[current_nearest == cluster], axis=0)

        last_nearest = current_nearest

    return clusters


def avg_iou(boxes, clusters):
    accuracy = np.mean([np.max(iou(boxes, clusters), axis=1)])
    return accuracy


def result2txt(data, fn='./yolo_anchors.txt'):
    f = open(fn, 'w')
    data = data.astype('uint')
    f.write(', '.join(f'{anchor[0]},{anchor[1]}' for anchor in data))
    f.close()
