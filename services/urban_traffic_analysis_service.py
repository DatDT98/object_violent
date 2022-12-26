import cv2
import numpy as np

from entities.common_entity import Area, Vehicle, TrafficViolationConfig, TrafficLightConfig, RoadLaneConfig, \
    VehicleLaneConfig
from entities.count_vehicle_response import CountedVehiclesByArea
from entities.parking_violation_response import ParkingViolationVehicleByArea
from services.core_services.vehicle_recognition_video_service import VehicleRecognitionVideoService, validate_area
from utils.application_properties import get_config_variable
from utils.common_function import find_object_location, pass_stop_line
from utils.deep_sort.track import Track
from utils.draw_results_to_frame import draw_vehicle_and_license_plate_frame, draw_counted_vehicles_debug_frame, \
    draw_parking_violation_debug_frame, draw_recognized_license_plate_frame, draw_violation_vehicle_to_frame


def is_red_light(frame, traffic_light):
    red_light = traffic_light.red_light
    if red_light.x1 == red_light.x2 == red_light.width == red_light.height == 0:
        return False
    red_light_matrix = frame[red_light.y1:red_light.y2, red_light.x1:red_light.x2]
    # green_light = traffic_light.green_light
    # green_light_matrix = frame[green_light.y1:green_light.y2, green_light.x1:green_light.x2]
    if np.mean(red_light_matrix[:, :, 2]) > 200:
        cv2.putText(frame, "Red Light", (0, 200), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        return True
    return False


def is_traffic_light_violation_vehicle(frame: np.ndarray, track: Track, traffic_light_config: TrafficLightConfig) -> bool:
    if track.is_traffic_light_violation:
        return True
    ignore_red_light_area = traffic_light_config.ignore_red_light_area
    area_id = find_object_location(track.bounding_box, [ignore_red_light_area])
    if area_id is None:
        return False
    traffic_light = traffic_light_config.traffic_light
    if is_red_light(frame, traffic_light) and len(track.center_list) > 0:
        track.red_light_center_list.append(track.center_list[-1])
        if pass_stop_line(track.red_light_center_list, traffic_light_config.stop_line):
            track.is_traffic_light_violation = True
            return True
    return False


def is_road_lane_violation_vehicle(track: Track, road_lane_config: RoadLaneConfig) -> bool:
    if track.is_road_lane_violation \
            or (find_object_location(track.bounding_box, road_lane_config.left_lane)
                and track.distance <= track.bounding_box.height)\
            or find_object_location(track.bounding_box, road_lane_config.right_lane) \
            and track.distance >= track.bounding_box.height:
        track.is_road_lane_violation = True
        return True
    return False


def is_vehicle_lane_violation_vehicle(track: Track, vehicle_lane_config: VehicleLaneConfig) -> bool:
    if track.is_vehicle_lane_violation:
        return True
    vehicle_type = track.cls
    if vehicle_type == "moto":
        area_id = find_object_location(track.bounding_box, vehicle_lane_config.other_lane)
        if area_id is not None:
            track.is_vehicle_lane_violation = True
            return True
    else:
        area_id = find_object_location(track.bounding_box, vehicle_lane_config.moto_lane)
        if area_id is not None:
            return True
    return False


def get_violation_vehicles_from_tracks(frame, tracks: [Track], traffic_violation_config: TrafficViolationConfig) \
        -> ([Vehicle], [Vehicle], [Vehicle]):
    traffic_light_violation_vehicles = []
    road_lane_violation_vehicles = []
    vehicle_lane_violation_vehicles = []
    for track in tracks:
        if not track.is_confirmed() or track.time_since_update > 0:
            continue
        vehicle_image = cv2.imencode(".jpg", track.vehicle_image)[1].tostring()
        recognized_vehicle = Vehicle(track_id=track.track_id,
                                     bounding_box=track.bounding_box,
                                     vehicle_type=track.cls,
                                     license_plate=track.license_plate_text,
                                     vehicle_image=vehicle_image)
        if is_traffic_light_violation_vehicle(frame, track, traffic_violation_config.traffic_light_config):
            traffic_light_violation_vehicles.append(recognized_vehicle)
        if is_road_lane_violation_vehicle(track, traffic_violation_config.road_lane_config):
            road_lane_violation_vehicles.append(recognized_vehicle)
        if is_vehicle_lane_violation_vehicle(track, traffic_violation_config.vehicle_lane_config):
            vehicle_lane_violation_vehicles.append(recognized_vehicle)
    return road_lane_violation_vehicles, traffic_light_violation_vehicles, vehicle_lane_violation_vehicles


class UrbanTrafficAnalysisService:
    def __init__(self, vehicle_recognition_video_service: VehicleRecognitionVideoService):
        self.vehicle_recognition_video_service = vehicle_recognition_video_service

    def recognize_license_plate(self, frame_generator, areas: [Area]):
        recognized_license_plate_generator = self.vehicle_recognition_video_service \
            .recognize_license_plate(frame_generator, areas)
        for license_plate_results, timestamp, frame in recognized_license_plate_generator:
            yield license_plate_results, timestamp, frame
            if get_config_variable("debug_mode"):
                show_frame = draw_recognized_license_plate_frame(frame, areas, license_plate_results)
                cv2.imshow("debug", show_frame)
                cv2.waitKey(1)

    def recognize_vehicle_and_license_plate(self, frame_generator):
        recognized_vehicles_generator = self.vehicle_recognition_video_service.recognize_vehicle_and_license_plate(
            frame_generator)
        for tracks, timestamp, frame in recognized_vehicles_generator:
            recognized_vehicles = []
            for track in tracks:
                if not track.is_confirmed() or track.time_since_update > 0:
                    continue
                vehicle_image = cv2.imencode(".jpg", track.vehicle_image)[1].tostring()
                recognized_vehicle = Vehicle(track_id=track.track_id,
                                             bounding_box=track.bounding_box,
                                             vehicle_type=track.cls,
                                             license_plate=track.license_plate_text,
                                             vehicle_image=vehicle_image)
                recognized_vehicles.append(recognized_vehicle)
            yield recognized_vehicles, timestamp, frame
            if get_config_variable("debug_mode"):
                show_frame = draw_vehicle_and_license_plate_frame(frame, recognized_vehicles)
                cv2.imshow("debug", show_frame)
                cv2.waitKey(1)

    def recognize_license_plate_video(self, frame_generator, areas: [Area]):
        yield from self.vehicle_recognition_video_service.recognize_license_plate(frame_generator, areas)

    def count_vehicles(self, frame_generator, areas: [Area]):
        counted_vehicles = {}
        for area in areas:
            counted_vehicles[area.area_id] = CountedVehiclesByArea(area.area_id)
        detected_vehicle_generator = self.vehicle_recognition_video_service.detect_vehicle(frame_generator)
        is_validated_area = False
        for tracks, timestamp, frame in detected_vehicle_generator:
            if not is_validated_area:
                validate_area(areas, frame.shape[0], frame.shape[1])
                is_validated_area = True
            for i, area in enumerate(areas):
                counted_vehicles[area.area_id].reset_current_bounding_boxes()
            for track in tracks:
                area_id = find_object_location(track.bounding_box, areas)
                if not track.is_confirmed() or track.time_since_update > 0 or area_id is None:
                    continue
                counted_vehicles[area_id].update_current_bounding_box(track.track_id, track.cls,
                                                                      track.bounding_box)
            yield counted_vehicles, timestamp

            if get_config_variable("debug_mode"):
                show_frame = draw_counted_vehicles_debug_frame(frame, areas, tracks, counted_vehicles)
                cv2.imshow("debug", show_frame)
                cv2.waitKey(1)

    def detect_parking_violation_vehicle(self, frame_generator, areas: [Area], parking_time_threshold: int):
        center_box_threshold = get_config_variable("center_box_threshold")
        recognized_vehicle_generator = self.vehicle_recognition_video_service \
            .recognize_vehicle_and_license_plate(frame_generator)
        is_validated_area = False
        for tracks, timestamp, frame in recognized_vehicle_generator:
            if not is_validated_area:
                validate_area(areas, frame.shape[0], frame.shape[1])
                is_validated_area = True
            parking_violation_vehicles = {}
            for area in areas:
                parking_violation_vehicles[area.area_id] = ParkingViolationVehicleByArea(area.area_id)
            for track in tracks:
                area_id = find_object_location(track.bounding_box, areas)
                if area_id is None or not track.is_confirmed():
                    continue
                center_box = track.get_center_box_within_time_range(parking_time_threshold)
                if center_box and center_box.width < center_box_threshold \
                        and center_box.height < center_box_threshold:
                    vehicle_image = cv2.imencode(".jpg", track.vehicle_image)[1].tostring()
                    parking_violation_vehicle = Vehicle(track_id=track.track_id,
                                                        bounding_box=track.bounding_box,
                                                        vehicle_type=track.cls,
                                                        vehicle_image=vehicle_image,
                                                        license_plate=track.license_plate_text)
                    parking_violation_vehicles[area_id].parking_violation_vehicles.append(parking_violation_vehicle)

            yield parking_violation_vehicles, timestamp
            if get_config_variable("debug_mode"):
                show_frame = draw_parking_violation_debug_frame(frame, areas, tracks, parking_violation_vehicles)
                cv2.imshow("debug", show_frame)
                cv2.waitKey(1)

    def detect_violation_vehicles(self, frame_generator, traffic_violation_config: TrafficViolationConfig):
        recognized_vehicles_generator = self.vehicle_recognition_video_service.recognize_vehicle_and_license_plate(
            frame_generator)
        for tracks, timestamp, frame in recognized_vehicles_generator:
            road_lane_violation_vehicles, traffic_light_violation_vehicles, vehicle_lane_violation_vehicles = \
                get_violation_vehicles_from_tracks(frame, tracks, traffic_violation_config)
            yield traffic_light_violation_vehicles, road_lane_violation_vehicles, \
                  vehicle_lane_violation_vehicles, timestamp
            print(timestamp)
            if get_config_variable("debug_mode"):
                show_frame = draw_violation_vehicle_to_frame(frame, tracks, traffic_violation_config)
                cv2.imshow("debug", show_frame)
                cv2.waitKey(1)
