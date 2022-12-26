import cv2
import numpy as np

from entities.common_entity import TrafficViolationConfig, Vehicle, Area, Box


def draw_vehicle(frame, bounding_box: Box, license_plate="", vehicle_type="",
                 box_color=(0, 225, 0), additional_text=None):
    cv2.rectangle(frame, (bounding_box.x1, bounding_box.y1),
                  (bounding_box.x2, bounding_box.y2), box_color, 3)
    cv2.putText(frame, license_plate,
                (bounding_box.x1, bounding_box.y1), cv2.FONT_HERSHEY_SIMPLEX, 2, box_color, 2)
    cv2.putText(frame, vehicle_type,
                (bounding_box.x1, bounding_box.y2), cv2.FONT_HERSHEY_SIMPLEX, 2, box_color, 2)

    if additional_text:
        cv2.putText(frame, additional_text,
                    (bounding_box.x1, bounding_box.y1 - int(bounding_box.height/5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, box_color, 2)


def draw_vehicle_and_license_plate_frame(frame, recognized_vehicles):
    frame_height, frame_width, _ = frame.shape
    for recognized_vehicle in recognized_vehicles:
        draw_vehicle(frame, recognized_vehicle.bounding_box, recognized_vehicle.license_plate,
                     recognized_vehicle.vehicle_type, (0, 225, 0))
    # cv2.imwrite("output_frames/" + str(int(round(time.time() * 1000))) + ".jpg", draw_frame)
    fx = 1
    if frame_width > 1920:
        fx = 1920 / frame_width
    return cv2.resize(frame, None, fx=fx, fy=fx)


def draw_recognized_license_plate_frame(frame, areas, license_plate_results):
    frame_height, frame_width, _ = frame.shape
    image_result = frame.copy()
    for area in areas:
        cv2.putText(image_result, "area: " + area.area_id, area.detection_area.get_left_bottom(),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, 255, 2)
        cv2.rectangle(image_result, area.detection_area.get_left_top(),
                      area.detection_area.get_right_bottom(), (0, 225, 0), 2)
    for license_plate in license_plate_results:
        cv2.putText(image_result, license_plate.license_plate,
                    license_plate.bounding_box.get_left_top(), cv2.FONT_HERSHEY_SIMPLEX, 2, 255, 2)
        cv2.rectangle(image_result, license_plate.bounding_box.get_left_top(),
                      license_plate.bounding_box.get_right_bottom(),
                      (0, 225, 0), 3)
        cv2.putText(image_result, str(license_plate.track_id),
                    license_plate.bounding_box.get_left_bottom(),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
    fx = 1
    if frame_width > 1920:
        fx = 1920 / frame_width
    return cv2.resize(image_result, None, fx=fx, fy=fx)


def draw_parking_violation_debug_frame(frame, areas, tracks, parking_violation_vehicles):
    image_result = frame.copy()
    frame_height, frame_width, _ = frame.shape
    for i, area in enumerate(areas):
        cv2.putText(image_result, "area: " + area.area_id, area.detection_area.get_left_bottom(),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, 255, 2)
        cv2.rectangle(image_result, area.detection_area.get_left_top(),
                      area.detection_area.get_right_bottom(), (0, 225, 0), 2)
    for track in tracks:
        if not track.is_confirmed() or track.time_since_update > 0:
            continue
        bounding_box = track.bounding_box
        cv2.rectangle(image_result, (int(bounding_box.x1), int(bounding_box.y1)),
                      (int(bounding_box.x2), int(bounding_box.y2)),
                      (0, 255, 0), 3)
        cv2.putText(image_result, str(track.track_id),
                    (int(bounding_box.x1), int(bounding_box.y2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)

        for center_point in track.center_list:
            cv2.circle(image_result, (center_point.x, center_point.y), radius=10, color=(0, 0, 255), thickness=-1)
    for area_id, parking_violation_vehicles_by_area in parking_violation_vehicles.items():
        for vehicle in parking_violation_vehicles_by_area.parking_violation_vehicles:
            cv2.putText(image_result, vehicle.license_plate,
                        vehicle.bounding_box.get_left_top(), cv2.FONT_HERSHEY_SIMPLEX, 2, 255, 2)
            cv2.rectangle(image_result,
                          vehicle.bounding_box.get_left_top(), vehicle.bounding_box.get_right_bottom(), (0, 0, 255), 3)
            cv2.putText(image_result, str(vehicle.track_id),
                        vehicle.bounding_box.get_left_bottom(),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
            cv2.putText(image_result, vehicle.license_plate,
                        vehicle.bounding_box.get_left_top(),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
    fx = 1
    if frame_width > 1920:
        fx = 1920 / frame_width
    return cv2.resize(image_result, None, fx=fx, fy=fx)


def draw_counted_vehicles_debug_frame(frame, areas, tracks, counted_vehicles):
    image_result = frame.copy()
    frame_height, frame_width, _ = frame.shape
    for i, area in enumerate(areas):
        counted_vehicle = counted_vehicles[area.area_id]
        cv2.rectangle(image_result, area.detection_area.get_left_top(),
                      area.detection_area.get_right_bottom(), (0, 225, 0), 2)
        text = ""
        for vehicle in counted_vehicle.detail:
            text += "{}: {}, ".format(vehicle.vehicle_type, vehicle.count)

        cv2.putText(image_result, "area: {}, total vehicles: {}, {}".format(area.area_id,
                                                                            len(counted_vehicle.track_ids),
                                                                            text),
                    area.detection_area.get_left_bottom(),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)

    for track in tracks:
        if not track.is_confirmed() or track.time_since_update > 0:
            continue
        bounding_box = track.bounding_box
        draw_vehicle(image_result, bounding_box=bounding_box,
                     license_plate="", vehicle_type=str(track.track_id),
                     box_color=(0, 225, 0), additional_text=None)
    fx = 1
    if frame_width > 1920:
        fx = 1920 / frame_width
    return cv2.resize(image_result, None, fx=fx, fy=fx)


def draw_contours_of_area(frame, area: Area, box_color=(0, 255, 255), additional_text=None):
    area_poly = area.poly
    contours = [(point.x, point.y) for point in area_poly]
    cv2.drawContours(frame, [np.array(contours)], 0, box_color, 2)
    cv2.putText(frame, additional_text,
                (int(area_poly[0].x), int(area_poly[0].y)),
                cv2.FONT_HERSHEY_SIMPLEX, 1, box_color, 2)


def draw_violation_vehicle_to_frame(frame, tracks, traffic_violation_config: TrafficViolationConfig):
    traffic_light_config = traffic_violation_config.traffic_light_config

    # draw stop line
    stop_line = traffic_light_config.stop_line
    if len(stop_line) > 1:
        contours = [(point.x, point.y) for point in stop_line]
        cv2.drawContours(frame, [np.array(contours)], 0, (0, 0, 255), 2)

    # draw light area
    traffic_light = traffic_light_config.traffic_light
    red_light = traffic_light.red_light
    cv2.rectangle(frame, (red_light.x1, red_light.y1), (red_light.x2, red_light.y2), (0, 0, 255), 3)
    green_light = traffic_light.green_light
    cv2.rectangle(frame, (green_light.x1, green_light.y1), (green_light.x2, green_light.y2), (0, 255, 0), 3)

    # draw ignore red light area
    ignore_red_light_area_poly = traffic_light_config.ignore_red_light_area.poly
    if len(ignore_red_light_area_poly) > 1:
        cv2.drawContours(frame, [np.array(ignore_red_light_area_poly)], 0, (0, 255, 0), 2)
        cv2.putText(frame, "ignore red light area",
                    (int(ignore_red_light_area_poly[0][0]), int(ignore_red_light_area_poly[0][1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # draw road lane config
    road_lane_config = traffic_violation_config.road_lane_config
    left_lane = road_lane_config.left_lane
    for area in left_lane:
        draw_contours_of_area(frame, area, (0, 255, 255), "left lane")

    right_lane = road_lane_config.right_lane
    for area in right_lane:
        draw_contours_of_area(frame, area, (255, 255, 0), "right lane")

    # draw vehicle lane config
    vehicle_lane_config = traffic_violation_config.vehicle_lane_config
    moto_lane = vehicle_lane_config.moto_lane
    for area in moto_lane:
        draw_contours_of_area(frame, area, (255, 0, 255), "moto lane")

    other_lane = vehicle_lane_config.other_lane
    for area in other_lane:
        draw_contours_of_area(frame, area, (255, 0, 0), "other lane")
    for track in tracks:
        if not track.is_confirmed() or track.time_since_update > 0:
            continue
        if track.is_road_lane_violation:
            box_color = (0, 0, 255)
            text = "road lane violation"
        elif track.is_vehicle_lane_violation:
            box_color = (0, 0, 255)
            text = "vehicle lane violation"
        elif track.is_traffic_light_violation:
            box_color = (0, 0, 255)
            text = "traffic light violation"
        else:
            box_color = (0, 255, 0)
            text = ""

        draw_vehicle(frame, bounding_box=track.bounding_box,
                     license_plate=track.license_plate_text,
                     vehicle_type=track.cls,
                     box_color=box_color,
                     additional_text=text)
        cv2.putText(frame, str(track.track_id), (track.bounding_box.x2, track.bounding_box.y1), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
        for center_point in track.center_list:
            cv2.circle(frame, (center_point.x, center_point.y), radius=10, color=(0, 0, 255), thickness=-1)

    frame_height, frame_width, _ = frame.shape
    fx = 1
    if frame_width > 1920:
        fx = 1920 / frame_width
    return cv2.resize(frame, None, fx=fx, fy=fx)
