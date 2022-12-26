from entities.common_entity import Point, Area, Box
import numpy as np
import cv2


def pass_stop_line(tracking_center_point_list, stop_line: [Point]) -> bool:
    if len(tracking_center_point_list) > 2:
        return not check_two_point_in_same_side(stop_line, tracking_center_point_list[0], tracking_center_point_list[-1])
    return False


def check_two_point_in_same_side(line: [Point], point1: Point, point2: Point):
    line_point_1 = line[0]
    line_point_2 = line[1]
    return ((line_point_1.y - line_point_2.y) * (point1.x - line_point_1.x) + (line_point_2.x - line_point_1.x) * (
            point1.y - line_point_1.y)) * ((line_point_1.y - line_point_2.y) * (point2.x - line_point_1.x) + (
            line_point_2.x - line_point_1.x) * (point2.y - line_point_1.y)) > 0


def find_object_location(bounding_box: Box, areas: [Area]):
    if bounding_box is None:
        return None
    center_x = int((bounding_box.x1 + bounding_box.x2) / 2)
    center_y = int((bounding_box.y1 + bounding_box.y2) / 2)
    for area in areas:
        area_bounding_box = area.detection_area
        if area_bounding_box \
                and area_bounding_box.x1 <= center_x <= area_bounding_box.x2 \
                and area_bounding_box.y1 < center_y < area_bounding_box.y2:
            return area.area_id
        if area.poly:
            contour = [(point.x, point.y) for point in area.poly]
            point = cv2.pointPolygonTest(np.array(contour), (center_x, center_y), False)
            if point > 0:
                return area.area_id
    return None
