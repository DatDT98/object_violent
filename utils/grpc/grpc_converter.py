from entities.common_entity import Area, Box, Point, LicensePlate, Vehicle, TrafficViolationConfig, TrafficLightConfig, \
    TrafficLight, RoadLaneConfig, VehicleLaneConfig
from entities.count_vehicle_response import CountedVehicle, CountedVehiclesByArea
from protos import traffic_server_pb2 as service


def convert_grpc_area_to_area(grpc_area) -> Area:
    return Area(area_id=grpc_area.area_id,
                detection_area=Box(x=grpc_area.detection_area.x,
                                   y=grpc_area.detection_area.y,
                                   width=grpc_area.detection_area.width,
                                   height=grpc_area.detection_area.height),
                poly=[Point(point.x, point.y) for point in grpc_area.poly])


def convert_grpc_areas_to_areas(grpc_areas) -> [Area]:
    return [convert_grpc_area_to_area(grpc_area) for grpc_area in grpc_areas]


def convert_license_plate_to_grpc(license_plate: LicensePlate) -> service.LicensePlateInfo:
    return service.LicensePlateInfo(track_id=str(license_plate.track_id),
                                    license_plate=license_plate.license_plate,
                                    area_id=license_plate.area_id,
                                    confidence=license_plate.confidence,
                                    license_plate_image=license_plate.license_plate_image)


def convert_license_plates_to_grpc(license_plates) -> [service.LicensePlateInfo]:
    return [convert_license_plate_to_grpc(license_plate) for license_plate in license_plates]


def convert_box_to_grpc_box(box: Box) -> service.Box:
    return service.Box(x=box.x1, y=box.y1, width=box.width, height=box.height)


def convert_grpc_box_to_box(box: service.Box) -> Box:
    return Box(x=box.x, y=box.y, width=box.width, height=box.height)


def convert_grpc_point_to_point(point: service.Point) -> Point:
    return Point(x=point.x, y=point.y)


def convert_grpc_points_to_points(points: [service.Point]) -> [Point]:
    return [Point(x=point.x, y=point.y) for point in points]


def convert_vehicle_to_grpc(vehicle: Vehicle) -> service.Vehicle:
    return service.Vehicle(track_id=str(vehicle.track_id),
                           vehicle_type=vehicle.vehicle_type,
                           license_plate=vehicle.license_plate,
                           bounding_box=convert_box_to_grpc_box(vehicle.bounding_box),
                           vehicle_image=vehicle.vehicle_image)


def convert_vehicles_to_grpc(vehicles: [Vehicle]) -> [service.Vehicle]:
    return [convert_vehicle_to_grpc(vehicle) for vehicle in vehicles]


def convert_counted_vehicle_to_grpc(counted_vehicle: CountedVehicle) -> service.CountedVehicles:
    current_bounding_boxes_grpc = [convert_box_to_grpc_box(bounding_box)
                                   for bounding_box in counted_vehicle.current_bounding_boxes]
    return service.CountedVehicles(vehicle_type=counted_vehicle.vehicle_type,
                                   count=counted_vehicle.count,
                                   current_bounding_boxes=current_bounding_boxes_grpc)


def convert_counted_vehicles_by_area_to_grpc(counted_vehicles_by_area: CountedVehiclesByArea) \
        -> service.CountedVehiclesByArea:
    grpc_total = len(counted_vehicles_by_area.track_ids)
    grpc_detail = [convert_counted_vehicle_to_grpc(counted_vehicle)
                   for counted_vehicle in counted_vehicles_by_area.detail]
    return service.CountedVehiclesByArea(area_id=counted_vehicles_by_area.area_id,
                                         total_vehicle=grpc_total, detail=grpc_detail)


def convert_counted_vehicles_per_frame_to_grpc(counted_vehicles_per_frame) -> [service.CountedVehiclesByArea]:
    return [convert_counted_vehicles_by_area_to_grpc(counted_vehicles_by_area)
            for area_id_response, counted_vehicles_by_area in counted_vehicles_per_frame.items()]


def convert_parking_violation_vehicles_to_grpc(parking_violation_vehicles_per_frame) -> [service.ParkingViolationByArea]:
    parking_violation_results = []
    for area_id, parking_violation_vehicles_by_area in parking_violation_vehicles_per_frame.items():
        parking_violation_vehicles_grpc = \
            convert_vehicles_to_grpc(parking_violation_vehicles_by_area.parking_violation_vehicles)
        parking_violation_results.append(
            service.ParkingViolationByArea(area_id=area_id,
                                           parking_violation_vehicles=parking_violation_vehicles_grpc))
    return parking_violation_results


def convert_grpc_traffic_light(grpc_traffic_light: service.TrafficLight) -> TrafficLight:
    red_light = convert_grpc_box_to_box(grpc_traffic_light.red_light)
    green_light = convert_grpc_box_to_box(grpc_traffic_light.green_light)
    return TrafficLight(red_light, green_light)


def convert_grpc_traffic_light_config(grpc_traffic_light_config: service.TrafficLightConfig) -> TrafficLightConfig:
    traffic_light = convert_grpc_traffic_light(grpc_traffic_light_config.traffic_light)
    stop_line = convert_grpc_points_to_points(grpc_traffic_light_config.stop_line)
    ignore_red_light_area = convert_grpc_area_to_area(grpc_traffic_light_config.ignore_red_light_area)
    return TrafficLightConfig(stop_line, traffic_light, ignore_red_light_area)


def convert_grpc_road_lane_config(grpc_road_lane_config: service.RoadLaneConfig) -> RoadLaneConfig:
    left_lane = convert_grpc_areas_to_areas(grpc_road_lane_config.left_lane)
    right_lane = convert_grpc_areas_to_areas(grpc_road_lane_config.right_lane)
    return RoadLaneConfig(left_lane, right_lane)


def convert_grpc_vehicle_lane_config(grpc_vehicle_lane_config: service.VehicleLaneConfig) -> VehicleLaneConfig:
    moto_lane = convert_grpc_areas_to_areas(grpc_vehicle_lane_config.moto_lane)
    other_lane = convert_grpc_areas_to_areas(grpc_vehicle_lane_config.other_lane)
    return VehicleLaneConfig(moto_lane, other_lane)


def convert_grpc_detect_violation_config(grpc_config: service.TrafficViolationConfig) -> TrafficViolationConfig:
    traffic_light_config = convert_grpc_traffic_light_config(grpc_config.traffic_light_config)
    road_lane_config = convert_grpc_road_lane_config(grpc_config.road_lane_config)
    vehicle_lane_config = convert_grpc_vehicle_lane_config(grpc_config.vehicle_lane_config)
    return TrafficViolationConfig(traffic_light_config, road_lane_config, vehicle_lane_config)
