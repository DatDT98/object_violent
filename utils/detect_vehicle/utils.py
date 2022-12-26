import cv2
import base64
import numpy as np

from utils import error_code
from utils.customized_exception import BadRequestException


def read_base64(base64_string):
    if len(base64_string) == 0:
        raise BadRequestException(error_code.IMAGE_BASE64_EMPTY, "image")
    if base64_string.startswith('data'):
        encoded_data = base64_string.split(',')[1]
    else:
        encoded_data = base64_string
    try:
        base64_string = base64.b64decode(encoded_data)
    except Exception:
        raise BadRequestException(error_code.INVALID_IMAGE_BASE64, 'image_base64')
    np_array = np.fromstring(base64_string, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    if img is None or len(img.shape) != 3 or img.shape[2] != 3:
        raise BadRequestException(error_code.INVALID_IMAGE_BASE64, 'image_base64')
    return img


def nms_cpu(boxes, confs, nms_thresh=0.5, min_mode=False):
    # print(boxes.shape)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = confs.argsort()[::-1]

    keep = []
    while order.size > 0:
        idx_self = order[0]
        idx_other = order[1:]

        keep.append(idx_self)

        xx1 = np.maximum(x1[idx_self], x1[idx_other])
        yy1 = np.maximum(y1[idx_self], y1[idx_other])
        xx2 = np.minimum(x2[idx_self], x2[idx_other])
        yy2 = np.minimum(y2[idx_self], y2[idx_other])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        if min_mode:
            over = inter / np.minimum(areas[order[0]], areas[order[1:]])
        else:
            over = inter / (areas[order[0]] + areas[order[1:]] - inter)

        inds = np.where(over <= nms_thresh)[0]
        order = order[inds + 1]

    return np.array(keep)


def post_processing(conf_thresh, nms_thresh, output):
    # anchors = [12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401]
    # num_anchors = 9
    # anchor_masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    # strides = [8, 16, 32]
    # anchor_step = len(anchors) // num_anchors

    # [batch, num, 1, 4]
    box_array = output[0]
    # [batch, num, num_classes]
    confs = output[1]

    if type(box_array).__name__ != 'ndarray':
        box_array = box_array.cpu().detach().numpy()
        confs = confs.cpu().detach().numpy()

    # [batch, num, 4]
    box_array = box_array[:, :, 0]

    # [batch, num, num_classes] --> [batch, num]
    max_conf = np.max(confs, axis=2)
    max_id = np.argmax(confs, axis=2)

    bboxes_batch = []
    for i in range(box_array.shape[0]):

        argwhere = max_conf[i] > conf_thresh
        l_box_array = box_array[i, argwhere, :]
        l_max_conf = max_conf[i, argwhere]
        l_max_id = max_id[i, argwhere]

        keep = nms_cpu(l_box_array, l_max_conf, nms_thresh)

        bboxes = []
        if (keep.size > 0):
            l_box_array = l_box_array[keep, :]
            l_max_conf = l_max_conf[keep]
            l_max_id = l_max_id[keep]

            for j in range(l_box_array.shape[0]):
                bboxes.append(
                    [l_box_array[j, 0], l_box_array[j, 1], l_box_array[j, 2], l_box_array[j, 3], l_max_conf[j],
                     l_max_conf[j], l_max_id[j]])

        bboxes_batch.append(bboxes)

    #     print('-----------------------------------')
    #     print('       max and argmax : %f' % (t2 - t1))
    #     print('                  nms : %f' % (t3 - t2))
    #     print('Post processing total : %f' % (t3 - t1))
    #     print('-----------------------------------')

    return bboxes_batch
