import numpy as np
import cv2
# from utils import face_align


def get_orthogonal_unit_vector(line):
    normalized = line / np.linalg.norm(line)
    unit_x = np.random.randn(2)
    unit_x -= unit_x.dot(normalized) * normalized
    unit_x /= np.linalg.norm(unit_x)
    return unit_x


def get_orthogonal_2nd_point(a, b, point):
    """
    get orthogonal vector of line (ab), crossing point
    :param a: np.array([x1, y1])
    :param b: np.array([x2, y2])
    :param point: np array [x_point, y_point]
    :return:
    """
    line = a - b
    return get_orthogonal_unit_vector(line) + point


def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        return None

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y


def spot_intersection(a, b, point):
    point_2 = get_orthogonal_2nd_point(a, b, point)
    line_1 = (a, b)
    line_2 = (point, point_2)
    return line_intersection(line_1, line_2)


def check_side_face_2(landmarks, threshold=5):
    left_eye = np.array([landmarks[5], landmarks[0]])
    right_eye = np.array([landmarks[6], landmarks[1]])
    nose = np.array([landmarks[7], landmarks[2]])
    left_mouth = np.array([landmarks[8], landmarks[3]])
    right_mouth = np.array([landmarks[9], landmarks[4]])

    if ((left_eye[0] < nose[0] and right_eye[0] < nose[0]) or (left_eye[0] > nose[0] and right_eye[0] > nose[0])) \
            and ((left_mouth[0] < nose[0] and right_mouth[0] < nose[0]) or (left_mouth[0] > nose[0] and right_mouth[0] > nose[0])):
        return True

    base_height = spot_intersection(left_eye, right_eye, nose)
    if base_height is None:
        return True
    base_height = np.array(base_height)
    left_distance = np.linalg.norm(base_height - left_eye)
    right_distance = np.linalg.norm(base_height - right_eye)
    if 1. / threshold < left_distance / right_distance < threshold:
        return False
    else:
        return True


def check_side_face(landmarks, threshold=4):
    """
    check side face
    :param landmarks: landmark of this face [y_left_eye, y_right_eye, y_nose, y_left_mouth, y_right_mouth,
     x_left_eye, x_right_eye, x_nose, x_left_mouth, x_right_mouth]
    :param threshold: threshold of (nose_x - left_eye_x) / (right_eye_x - nose_x)
    :return: boolean: is side face
    """
    left_eye_x = landmarks[5]
    right_eye_x = landmarks[6]
    nose_x = landmarks[7]

    # print((nose_x - left_eye_x) / (right_eye_x - nose_x))
    if right_eye_x < nose_x or left_eye_x > nose_x:
        return True
    if nose_x == left_eye_x or right_eye_x == nose_x:
        return True
    if not (1 / threshold < (nose_x - left_eye_x) / (right_eye_x - nose_x) < threshold):
        return True

    return False


def is_rotated_face(landmarks):
    """
    check face is rotated or not
    :param landmarks: landmark of this face [y_left_eye, y_right_eye, y_nose, y_left_mouth, y_right_mouth,
     x_left_eye, x_right_eye, x_nose, x_left_mouth, x_right_mouth]
    :return: boolean
    """
    return landmarks[2] < landmarks[0] or landmarks[2] < landmarks[1]


def variance_of_laplacian(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def check_face_small(rectangle):
    min_w = abs(rectangle[2] - rectangle[0])
    min_h = abs(rectangle[3] - rectangle[1])
    return min(min_h, min_w)


def get_coordinates_with_margin(image, bbox):
    margin = (bbox[2] - bbox[0]) / 2
    x1 = max(0, bbox[0] - margin / 2)
    x2 = min(image.shape[1], bbox[2] + margin / 2)
    y1 = max(0, bbox[1] - margin / 2)
    y2 = min(image.shape[0], bbox[3] + margin / 2)
    return x1, y1, x2, y2


def crop_image_to_save(image, bbox):
    x1, y1, x2, y2 = get_coordinates_with_margin(image, bbox)
    crop_img = image[int(y1):int(y2), int(x1):int(x2)]
    return crop_img


def convert_bbox_to_x_y_w_h(image, bbox):
    x1, y1, x2, y2 = get_coordinates_with_margin(image, bbox)
    return [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]



def get_face_in_center(img_size, bounding_boxes):
    """
    get face in center of image
    :param img_size: image size (height, width, channel)
    :param bounding_boxes: bounding box
    :return: index of center face
    """
    bounding_box_size = (bounding_boxes[:, 2] - bounding_boxes[:, 0]) * (bounding_boxes[:, 3] - bounding_boxes[:, 1])
    img_center = img_size / 2
    offsets = np.vstack(
        [(bounding_boxes[:, 0] + bounding_boxes[:, 2]) / 2 - img_center[1],
         (bounding_boxes[:, 1] + bounding_boxes[:, 3]) / 2 - img_center[0]])
    offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
    index = np.argmax(bounding_box_size - offset_dist_squared * 2.0)  # some extra weight on the centering
    return index


def get_face_to_check_anti_spoofing(image):
    height, width, _ = image.shape
    new_height = height / 1.4 * 1.25
    new_width = width / 1.4 * 1.25
    margin_height = int((height - new_height) / 2)
    margin_width = int((width - new_width) / 2)
    return image.copy()[margin_height: height - margin_height, margin_width: width - margin_width, ]


def clip_box(im_height, im_width, box):
    """
    Inplace clip box to [0, im_height / im_width]
    Args:
        im_height (int): maximum height
        im_width (int): maximum width
        box: list or numpy array with shape (4,) [x1, y1, x2, y2]

    Returns:
        clipped_box: np
    """
    box[0::2] = np.clip(box[0::2], 0, im_width)
    box[1::2] = np.clip(box[1::2], 0, im_height)
    return box


def crop_face_image_with_margin(raw_frame, predicted_box, expand_ratio=1.0):
    """
    Return expanded image from raw_frame, do nothing if expand_ratio = 1.0
    Args:
        raw_frame: np array image with shape (-1, -1, 3)
        predicted_box: integer list or numpy array [x1, y1, x2, y2]
        expand_ratio: expanded ratio > 1

    Returns:

    """
    assert expand_ratio >= 1., "Expect expand_ratio > 1, but got: {}".format(expand_ratio)

    x1, y1, x2, y2 = predicted_box
    if expand_ratio == 1.0:
        face_image = raw_frame[y1: y2+1, x1: x2+1]
        return face_image, x1, y1

    new_width = (x2 - x1) / 2 * expand_ratio
    new_height = (y2 - y1) / 2 * expand_ratio
    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2
    new_top = max(0, int(y_center - new_height))
    new_bottom = min(raw_frame.shape[0], int(y_center + new_height))
    new_left = max(0, int(x_center - new_width))
    new_right = min(raw_frame.shape[1], int(x_center + new_width))
    face_image = raw_frame[new_top:new_bottom, new_left:new_right]
    return face_image, new_left, new_top
