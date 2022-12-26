from entities.count_vehicle_response import CountedVehiclesByArea
from services.core_services.tracking_service import TrackingService, init_tracker
from services.core_services.vehicle_recognition_image_service import VehicleRecognitionImageService
import numpy as np
import cv2

from utils.deep_sort import preprocessing
from utils import error_code
from utils.application_properties import get_config_variable
from utils.customized_exception import BadRequestException
from utils.deep_sort.detection import Detection
from entities.common_entity import Area, LicensePlate, Box
from utils.draw_results_to_frame import draw_recognized_license_plate_frame


def get_image_from_box(bounding_box: Box, frame):
    frame_height, frame_width, _ = frame.shape
    x_min = max(0, int(bounding_box.x1))
    y_min = max(0, int(bounding_box.y1))
    x_max = min(frame_width, int(bounding_box.x2))
    y_max = min(frame_height, int(bounding_box.y2))
    return frame[y_min: y_max, x_min: x_max]


def count_vehicles_from_tracks(counted_vehicles_by_area, tracker):
    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 0:
            continue
        counted_vehicles_by_area.update_current_bounding_box(track.cls, track.bounding_box)
        if not track.is_counted:
            counted_vehicles_by_area.increase_vehicle(track.cls)
            track.is_counted = True


def validate_area(areas, frame_height, frame_width):
    for i, area in enumerate(areas):
        x_area_min, y_area_min, x_area_max, y_area_max = area.detection_area.get_left_top_right_bottom()
        if not (0 <= x_area_min < x_area_max and x_area_min < x_area_max < frame_width
                and 0 <= y_area_min < y_area_max and y_area_min < y_area_max < frame_height):
            raise BadRequestException(error_code.INVALID_AREA, "areas", x_area_min, y_area_min,
                                      area.detection_area.width, area.detection_area.height)


def init_vehicle_results_dict_by_area(areas):
    result_dict = {}
    for area in areas:
        result_dict[area.area_id] = []
    return result_dict


def parse_point_plate_to_tracker_input(point_plate_list, area):
    if point_plate_list[0] is None:
        tracker_input = np.empty((0, 5))
    else:
        tracker_input = np.array(point_plate_list) \
                         + np.array([area.detection_area.x1, area.detection_area.y1,
                                     area.detection_area.x1, area.detection_area.y1, 0])
    return tracker_input


class VehicleRecognitionVideoService:
    def __init__(self, vehicle_recognition_image_service: VehicleRecognitionImageService, tracking_service: TrackingService):
        self.vehicle_recognition_image_service = vehicle_recognition_image_service
        self.tracking_service = tracking_service
        self.nms_max_overlap = 1.0

    def __detect_and_track_vehicles(self, frame, tracker):
        bounding_boxes, confidences, label = self.vehicle_recognition_image_service.detect_vehicle(frame.copy())
        feature_vectors = self.tracking_service.extract_tracking_feature(frame, bounding_boxes)
        detections = [Detection(bbox, confidence, cls, feature) for bbox, confidence, cls, feature in
                      zip(bounding_boxes, confidences, label, feature_vectors)]
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, self.nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        tracker.predict()
        tracker.update(detections)

    def recognize_license_plate(self, frame_generator, areas: [Area]):
        trackers = []
        recognized_license_plates = []
        is_validated_area = False
        for i, area in enumerate(areas):
            recognized_license_plates.append(CountedVehiclesByArea(area.area_id))
            trackers.append(init_tracker("sort"))
        for frame, timestamp in frame_generator:
            license_plate_results = []
            frame_height, frame_width, _ = frame.shape

            for i, area in enumerate(areas):
                # validate area config
                if not is_validated_area:
                    validate_area(areas, frame_height, frame_width)
                    is_validated_area = True

                image = get_image_from_box(area.detection_area, frame)
                point_plate_list, license_plate_images = self.vehicle_recognition_image_service\
                    .detect_one_license_plate_per_image([image])
                tracker_input = parse_point_plate_to_tracker_input(point_plate_list, area)
                trackers[i].update(tracker_input)
                # expect only one license plate per area
                if license_plate_images[0] is not None and trackers[i].trackers:
                    track = trackers[i].trackers[0]
                    self.__update_license_plate_info_of_track(track, license_plate_images)
                    bounding_box = Box(x=int(tracker_input[0, 0]), y=int(tracker_input[0, 1]),
                                       width=int(tracker_input[0, 2] - tracker_input[0, 0]),
                                       height=int(tracker_input[0, 3] - tracker_input[0, 1]))
                    license_plate_results.append(LicensePlate(track_id=track.id,
                                                              area_id=area.area_id,
                                                              license_plate=track.license_plate,
                                                              confidence=track.confidence,
                                                              bounding_box=bounding_box,
                                                              license_plate_image=track.license_plate_image))
            yield license_plate_results, timestamp, frame
            if get_config_variable("debug_mode"):
                show_frame = draw_recognized_license_plate_frame(frame, areas, license_plate_results)
                cv2.imshow("debug", show_frame)
                cv2.waitKey(1)

    def __update_license_plate_info_of_track(self, track, license_plate_images):
        if track.confidence < 95:
            labels, probs = self.vehicle_recognition_image_service \
                .extract_text_from_license_plates(license_plate_images)
            if track.confidence <= probs[0]:
                track.license_plate = labels[0]
                track.license_plate_image = cv2.imencode('.jpg', license_plate_images[0])[1].tostring()
                track.confidence = probs[0]

    def detect_vehicle(self, frame_generator):
        tracker = init_tracker("deep_sort")
        # used to record the time when we processed last frame
        for frame, timestamp in frame_generator:
            frame_height, frame_width, _ = frame.shape
            self.__detect_and_track_vehicles(frame, tracker)
            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 0:
                    continue
                track.vehicle_image = get_image_from_box(track.bounding_box, frame)
            yield tracker.tracks, timestamp, frame

    def recognize_vehicle_and_license_plate(self, frame_generator):
        recognized_vehicles_generator = self.detect_vehicle(frame_generator)
        for tracks, timestamp, frame in recognized_vehicles_generator:
            for track in tracks:
                if not track.is_confirmed() or track.time_since_update > 0:
                    continue
                if track.vehicle_image is not None and track.license_plate_text == '':
                    track.license_plate_text = self.vehicle_recognition_image_service\
                        .extract_license_plate_from_vehicle_image(track.vehicle_image)
            yield tracks, timestamp, frame
