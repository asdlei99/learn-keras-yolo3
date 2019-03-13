# coding: utf-8
import cv2
import numpy as np

from nms import non_max_suppression


def _get_anchors(fn='./model_data/anchors.txt'):
    anchors = [int(x) for x in open(fn).read().split(',')]
    # batch, num_anchors, box_params
    return np.array(anchors).reshape(-1, 3, 2)


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def yolo_correct_boxes(box_xy, box_hw):
    box_mins = box_xy - (box_hw / 2.)
    box_maxes = box_xy + (box_hw / 2.)
    return np.concatenate((box_mins, box_maxes), axis=-1)


def yolo_eval(outputs, anchors, score_threshold=.6, iou_threshold=.5):
    boxes_ = []
    scores_ = []
    classes_ = []

    h, w = outputs[0].shape[1:3]
    size = (h * 32, w * 32)

    for output, anchor in zip(outputs, anchors):
        # 整理结果
        n = output.shape[0]
        grid_shape = output.shape[1:3]  # height, width

        # 变维：个数，高度，宽度，锚数，目标（位置，大小，置信度，类别）
        output.shape = (n, grid_shape[0], grid_shape[1], 3, -1)
        # 计算准确坐标
        box_hw = np.exp(output[..., 2:4]) * anchor
        box_confidence = sigmoid(output[..., 4:5])
        box_class_probs = sigmoid(output[..., 5:])
        box_scores = box_confidence * box_class_probs

        mask = box_scores > score_threshold
        for c in range(mask.shape[-1]):
            w = np.where(mask[..., c])
            box_xy = (sigmoid(output[..., :2][w]) + np.array(w[1:-1]).T) / grid_shape * size
            _boxes = yolo_correct_boxes(box_xy, box_hw[w]).astype('int')
            _scores = box_scores[..., c][w]
            pick = non_max_suppression(_boxes, iou_threshold, _scores)
            boxes_.extend(_boxes[pick])
            scores_.extend(_scores[pick])
            classes = np.ones_like(pick, 'uint') * c
            classes_.extend(classes)
    return boxes_, scores_, classes_


def main():
    # 加载图片
    im = cv2.imread('./images/416x416.jpg')
    # 加载框架的原始输出
    data = np.load('./outputs/416x416.jpg.output.npz')
    outputs = [data[f'output{i}'] for i in range(3)]
    # 加载锚框
    anchors = _get_anchors()

    boxes, scores, classes = yolo_eval(outputs, anchors)

    # loop over the picked bounding boxes and draw them
    print(scores, classes)
    for (start_x, start_y, end_x, end_y) in boxes:
        cv2.rectangle(im, (start_y, start_x), (end_y, end_x), (0, 255, 0), 2)
    # display the images
    cv2.imshow("After NMS", im)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
