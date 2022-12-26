# vim: expandtab:ts=4:sw=4
import time
from collections import Counter

from entities.common_entity import Box, Point
from utils.deep_sort.detection import Detection


class TrackState:
    """
    Enumeration type for the single target track state. Newly created tracks are
    classified as `tentative` until enough evidence has been collected. Then,
    the track state is changed to `confirmed`. Tracks that are no longer alive
    are classified as `deleted` to mark them for removal from the set of active
    tracks.

    """

    TENTATIVE = 1
    CONFIRMED = 2
    DELETED = 3


class Track:
    """
    A single target track with state space `(x, y, a, h)` and associated
    velocities, where `(x, y)` is the center of the bounding box, `a` is the
    aspect ratio and `h` is the height.

    Parameters
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.
    max_age : int
        The maximum number of consecutive misses before the track state is
        set to `Deleted`.
    feature : Optional[ndarray]
        Feature vector of the detection this track originates from. If not None,
        this feature is added to the `features` cache.

    Attributes
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    hits : int
        Total number of measurement updates.
    age : int
        Total number of frames since first occurance.
    time_since_update : int
        Total number of frames since last measurement update.
    state : TrackState
        The current track state.
    features : List[ndarray]
        A cache of features. On each measurement update, the associated feature
        vector is added to this list.

    """

    def __init__(self, mean, covariance, track_id, n_init, max_age, adc_threshold,
                 detection: Detection=None):
        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.det_cls = detection.cls  # the class from detection
        self.counter = Counter()
        self.cls = None  # for most common class

        self.total_prob = 0
        self.adc_threshold = adc_threshold  # Average detection confidence threshold
        self.detection_confidence = detection.confidence
        self.adc = 0

        self.state = TrackState.TENTATIVE
        self.features = []
        if detection is not None:
            self.features.append(detection.feature)
        self.license_plate_text = ''
        self.vehicle_image = None

        self._n_init = n_init
        self._max_age = max_age
        top_left_x, top_left_y, bottom_right_x, bottom_right_y = detection.to_tlbr()
        self.bounding_box = Box(top_left_x, top_left_y, bottom_right_x - top_left_x,
                                bottom_right_y - top_left_y)
        self.is_counted = False
        self.init_time = time.time()
        self.current_time = time.time()
        self.time_list = []
        self.center_list: [Point] = []
        self.is_road_lane_violation = False
        self.is_vehicle_lane_violation = False
        self.is_traffic_light_violation = False
        self.distance = 0  # distance of vehicle
        self.red_light_center_list: [Point] = []

    def to_tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
        width, height)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    def to_tlbr(self):
        """Get current position in bounding box format `(min x, miny, max x,
        max y)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret

    def predict(self, kf):
        """Propagate the state distribution to the current time step using a
        Kalman filter prediction step.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.

        """
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(self, kf, detection):
        """Perform Kalman filter measurement update step and update the feature
        cache.

        This version creates tracks only when the average detection confidence is
        higher than the set threshold.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.
        detection : Detection
            The associated detection.

        """
        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, detection.to_xyah())
        self.features.append(detection.feature)
        self.counter[self.det_cls] += 1
        self.cls = self.counter.most_common(1)[0][0]  # get most common cls for track

        self.hits += 1
        self.time_since_update = 0
        self.total_prob += self.detection_confidence
        self.adc = self.total_prob / self.hits
        if self.state == TrackState.TENTATIVE and self.hits >= self._n_init:
            if self.adc < self.adc_threshold:
                self.state = TrackState.DELETED
            else:
                self.state = TrackState.CONFIRMED

        top_left_x, top_left_y, bottom_right_x, bottom_right_y = detection.to_tlbr()
        self.bounding_box = Box(top_left_x, top_left_y, bottom_right_x - top_left_x,
                                bottom_right_y - top_left_y)
        self.current_time = time.time()
        if self.hits % 1 == 0:
            self.time_list.append(time.time())
            self.center_list.append(Point(int((top_left_x + bottom_right_x) / 2), int((top_left_y + bottom_right_y) / 2)))
            self.update_distance()

    def update_distance(self):
        center_list_length = len(self.center_list)
        if center_list_length > 1:
            last_distance = self.center_list[-1].y - self.center_list[-2].y
            if last_distance * self.distance >= 0:
                self.distance += last_distance
            else:
                self.distance = last_distance

    def mark_missed(self):
        """Mark this track as missed (no association at the current time step).
        """
        if self.state == TrackState.TENTATIVE or self.time_since_update > self._max_age:
            self.state = TrackState.DELETED

    def is_tentative(self):
        """Returns True if this track is tentative (unconfirmed).
        """
        return self.state == TrackState.TENTATIVE

    def is_confirmed(self):
        """Returns True if this track is confirmed."""
        return self.state == TrackState.CONFIRMED

    def is_deleted(self):
        """Returns True if this track is dead and should be deleted."""
        return self.state == TrackState.DELETED

    def get_center_box_within_time_range(self, time_threshold):
        current_time = time.time()
        if not self.time_list or current_time - self.time_list[0] < time_threshold:
            return None
        min_index = 0
        for i, time_element in enumerate(self.time_list):
            if current_time - time_element < time_threshold:
                min_index = i
                break

        center_list_at_time = self.center_list[min_index:len(self.center_list)]
        x_min = 99999
        y_min = 99999
        x_max = 0
        y_max = 0
        for center_point in center_list_at_time:
            x, y = center_point.x, center_point.y
            if x < x_min:
                x_min = x
            if y < y_min:
                y_min = y

            if x > x_max:
                x_max = x
            if y > y_max:
                y_max = y

        return Box(x=x_min, y=y_min, width=x_max - x_min, height=y_max - y_min)
