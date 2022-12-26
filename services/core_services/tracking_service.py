import cv2
import numpy as np

from services.core_services.serving_service import ServingService
from utils.application_properties import get_config_variable
# from utils.logging import logger
from utils.deep_sort.tracker import Tracker
from utils.deep_sort import nn_matching
from utils.sort_tracking import sort


def init_tracker(tracker_type):
    if tracker_type == "deep_sort":
        metric = nn_matching.NearestNeighborDistanceMetric(
            metric="cosine", matching_threshold=0.3, budget=None)
        return Tracker(metric, max_age=get_config_variable("track_max_age"))
    elif tracker_type == "sort":
        return sort.Sort(max_age=get_config_variable("track_max_age"))
    else:
        raise BaseException("Only accept tracker_type 'deep_sort' or 'sort'.")


def extract_image_patch(image, bbox, patch_shape):
    """Extract image patch from bounding box.

    Parameters
    ----------
    image : ndarray
        The full image.
    bbox : array_like
        The bounding box in format (x, y, width, height).
    patch_shape : Optional[array_like]
        This parameter can be used to enforce a desired patch shape
        (height, width). First, the `bbox` is adapted to the aspect ratio
        of the patch shape, then it is clipped at the image boundaries.
        If None, the shape is computed from :arg:`bbox`.

    Returns
    -------
    ndarray | NoneType
        An image patch showing the :arg:`bbox`, optionally reshaped to
        :arg:`patch_shape`.
        Returns None if the bounding box is empty or fully outside of the image
        boundaries.

    """
    bbox = np.array(bbox)
    if patch_shape is not None:
        # correct aspect ratio to patch shape
        target_aspect = float(patch_shape[1]) / patch_shape[0]
        new_width = target_aspect * bbox[3]
        bbox[0] -= (new_width - bbox[2]) / 2
        bbox[2] = new_width

    # convert to top left, bottom right
    bbox[2:] += bbox[:2]
    bbox = bbox.astype(np.int)

    # clip at image boundaries
    bbox[:2] = np.maximum(0, bbox[:2])
    bbox[2:] = np.minimum(np.asarray(image.shape[:2][::-1]) - 1, bbox[2:])
    if np.any(bbox[:2] >= bbox[2:]):
        return None
    sx, sy, ex, ey = bbox
    image = image[sy:ey, sx:ex]
    image = cv2.resize(image, tuple(patch_shape[::-1]))
    return image


class TrackingService:
    def __init__(self, serving_service: ServingService):
        self.serving_service = serving_service
        self.tracking_feature_dim = 128
        self.tracking_image_shape = [128, 64, 3]

    def extract_tracking_feature(self, image, bounding_boxes):
        image_patches = []
        for box in bounding_boxes:
            patch = extract_image_patch(image, box, self.tracking_image_shape[:2])
            if patch is None:
                # logger.debug("WARNING: Failed to extract image patch: %s." % str(box))
                patch = np.random.uniform(
                    0., 255., self.tracking_image_shape).astype(np.uint8)
            image_patches.append(patch)
        image_patches = np.asarray(image_patches)
        return self.serving_service.extract_tracking_features(image_patches)