import numpy as np
import cv2
from utils.plate_detection.label import Label


def im2single(image):
    assert (image.dtype == 'uint8')
    return image.astype('float32') / 255.


def get_wh(shape):
    return np.array(shape[1::-1]).astype(float)


def iou_labels(l1, l2):
    return iou(l1.tl(), l1.br(), l2.tl(), l2.br())


def nms(labels, iou_threshold=.5):
    selected_labels = []
    labels.sort(key=lambda l: l.prob(), reverse=True)

    for label in labels:

        non_overlap = True
        for sel_label in selected_labels:
            if iou_labels(label, sel_label) > iou_threshold:
                non_overlap = False
                break

        if non_overlap:
            selected_labels.append(label)

    return selected_labels


def hsv_transform(image, hsv_modifier):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image = image + hsv_modifier
    return cv2.cvtColor(image, cv2.COLOR_HSV2BGR)


def iou(tl1, br1, tl2, br2):
    wh1, wh2 = br1 - tl1, br2 - tl2
    assert ((wh1 >= .0).all() and (wh2 >= .0).all())

    intersection_wh = np.maximum(np.minimum(br1, br2) - np.maximum(tl1, tl2), 0.)
    intersection_area = np.prod(intersection_wh)
    area1, area2 = (np.prod(wh1), np.prod(wh2))
    union_area = area1 + area2 - intersection_area
    return intersection_area / union_area


class DLabel(Label):
    def __init__(self, cl, pts, prob):
        self.pts = pts
        tl = np.amin(pts, 1)
        br = np.amax(pts, 1)
        Label.__init__(self, cl, tl, br, prob)

    def add_pad(self, pad_percent=0.1):
        w, h = self.wh()
        pad_w = w * pad_percent / 2
        pad_h = 0
        pad_matrix = [[-pad_w, pad_w, pad_w, -pad_w], [-pad_h, -pad_h, pad_h, pad_h]]
        self.pts += pad_matrix


def find_t_matrix(pts, t_pts):
    A = np.zeros((8, 9))
    for i in range(0, 4):
        xi = pts[:, i]
        xil = t_pts[:, i]
        xi = xi.T

        A[i * 2, 3:6] = -xil[2] * xi
        A[i * 2, 6:] = xil[1] * xi
        A[i * 2 + 1, :3] = xil[2] * xi
        A[i * 2 + 1, 6:] = -xil[0] * xi

    [U, S, V] = np.linalg.svd(A)
    H = V[-1, :].reshape((3, 3))

    return H


def get_rect_points(tlx, tly, brx, bry):
    return np.array([[tlx, brx, brx, tlx], [tly, tly, bry, bry], [1., 1., 1., 1.]], dtype=float)


def decode_predict(license_plate_response_matrix, resized_image_shape, threshold=.9):
    net_stride = 2 ** 4
    side = ((208. + 40.) / 2.) / net_stride  # 7.75

    probs = license_plate_response_matrix[..., 0]
    affines = license_plate_response_matrix[..., 2:]
    rx, ry = license_plate_response_matrix.shape[:2]

    #     print("prob maximum:",np.max(Probs))
    xx, yy = np.where(probs > threshold)

    WH = get_wh(resized_image_shape)
    MN = WH / net_stride

    vxx = vyy = 0.5  # alpha

    base = lambda vx, vy: np.matrix(
        [[-vx, -vy, 1.], [vx, -vy, 1.], [vx, vy, 1.], [-vx, vy, 1.]]).T
    labels = []

    for i in range(len(xx)):
        y, x = xx[i], yy[i]
        affine = affines[y, x]
        prob = probs[y, x]

        mn = np.array([float(x) + .5, float(y) + .5])

        A = np.reshape(affine, (2, 3))
        A[0, 0] = max(A[0, 0], 0.)
        A[1, 1] = max(A[1, 1], 0.)

        pts = np.array(A * base(vxx, vyy))  # *alpha
        pts_mn_center_mn = pts * side
        pts_mn = pts_mn_center_mn + mn.reshape((2, 1))

        pts_prop = pts_mn / MN.reshape((2, 1))

        labels.append(DLabel(0, pts_prop, prob))

    #     print("number of prediction: ",len(labels))
    final_labels = nms(labels, .1)
    #     print("the number after nms:",len(final_labels))

    return final_labels


def reconstruct(original_image, final_labels):
    h, w, _ = original_image.shape
    final_labels.sort(key=lambda x: x.prob(), reverse=True)
    # One plate per vehicle
    label = final_labels[0]
    out_size = (np.int(label.wh()[0] * w), np.int(label.wh()[1] * h))
    t_ptsh = get_rect_points(0, 0, out_size[0], out_size[1])

    ptsh = np.concatenate((label.pts * get_wh(original_image.shape).reshape((2, 1)), np.ones((1, 4))))
    H = find_t_matrix(ptsh, t_ptsh)
    license_plate_image = cv2.warpPerspective(original_image, H, out_size, borderValue=.0, flags=cv2.INTER_CUBIC)
    return license_plate_image, label
