from concurrent import futures

import cv2
import grpc
from models.experimental import attempt_load
from utils import config, error_code
from tracker.sort import KalmanBoxTracker, Sort
from utils.customized_exception import BadRequestException
from utils.video_source_reader import VideoSourceReader
from detector import violate_object_detect, leave_object_detect
from protos import object_server_pb2 as service
from protos import object_server_pb2_grpc
from utils.torch_utils import select_device
from utils.logging import setup_logging_json
import logging
import os
from protos.object_server_pb2 import (
    DetectObjResponse
)
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'

LOGGER = logging.getLogger(__name__)
def get_api_key(context) -> str:
    """
    Get api_key in metadata, raise error if does not exist
    Args:
        context: gRPC context
    Returns:
        api_key: (str) key to use API
    """
    provided_api_key = ""
    for key, value in context.invocation_metadata():
        if key == "api_key":
            provided_api_key = str(value)
            return provided_api_key
    return provided_api_key

class ObjectService(object_server_pb2_grpc.ObjectServiceServicer):

    def LeaveObjectDetect(self, request, context):
        if request.ws is None:
            context.abort(grpc.StatusCode.NOT_FOUND, "WebSocket not found")
            LOGGER.error("WebSocket not found")
        try:
            url = request.ws
            api_key = get_api_key(context)
            if api_key == "":
                raise BadRequestException(error_code.UNAUTHORIZED, "api_key")
            LOGGER.info("Reading data from url: {}".format(url))
            reader = VideoSourceReader(source_url=url)
            tracker = Sort()
            objects = leave_object_detect(model, reader.frame_generator, image_size, iou_thres, conf_thres, device, tracker,s, msg)
            for timestamp, trackers, frame in objects:
                obj = []
                image_bytes = None

                for tracker in trackers:
                    box = service.Box(x=tracker.bbox[0], y=tracker.bbox[1],
                                      width=tracker.bbox[2] - tracker.bbox[0],
                                      height=tracker.bbox[3] - tracker.bbox[1])
                    obj.append(service.Obj(bounding_box=box, confidence=tracker.confidence,
                                                  track_id=tracker.track_id))
                    _, buffer = cv2.imencode(".jpg", frame)
                    image_bytes = (buffer.tobytes())
                obj_image = image_bytes
                if obj is not None:
                    yield DetectObjResponse(forgot_obj=obj, timestamp=timestamp, image_bytes = obj_image)
        except Exception as e:
            LOGGER.error(e)
            yield DetectObjResponse()


    def ViolateObjectDetect(self, request, context):
        if request.ws is None:
            context.abort(grpc.StatusCode.NOT_FOUND, "WebSocket not found")
            LOGGER.error("WebSocket not found")
        try:
            url = request.ws
            LOGGER.info("Reading data from url: {}".format(url))
            tracker = Sort()
            alert_time = []
            list_track = []
            areas = request.areas
            api_key = get_api_key(context)
            if api_key == "":
                raise BadRequestException(error_code.UNAUTHORIZED, "api_key")
            reader = VideoSourceReader(source_url=url)

            objects = violate_object_detect(model, reader.frame_generator, tracker, areas, alert_time, list_track)
            for timestamp, trackers, frame, is_send in objects:
                obj = []
                image_bytes = None

                for tracker in trackers:
                    box = service.Box(x=tracker.bbox[0], y=tracker.bbox[1],
                                      width=tracker.bbox[2] - tracker.bbox[0],
                                      height=tracker.bbox[3] - tracker.bbox[1])
                    obj.append(service.Obj(bounding_box=box, confidence=tracker.confidence,
                                           track_id=tracker.track_id))
                    _, buffer = cv2.imencode(".jpg", frame)
                    image_bytes = (buffer.tobytes())
                obj_image = image_bytes
                if obj is not None:
                    yield DetectObjResponse(forgot_obj=obj, timestamp=timestamp, image_bytes=obj_image, send=is_send)
        except Exception as e:
            LOGGER.error(e)
            yield DetectObjResponse()
def serve():
    # Initialize server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    object_server_pb2_grpc.add_ObjectServiceServicer_to_server(
        ObjectService(), server
    )
    server.add_insecure_port("[::]:50051")
    server.start()
    print("Starting...")
    server.wait_for_termination()


if __name__ == "__main__":
    setup_logging_json("logging.json")
    weights = config.get_model()
    image_size = int(config.get_image_size())
    iou_thres = float(config.get_iou_threshold())
    conf_thres = float(config.get_confidence_threshold())
    # get device from file config
    device = config.get_divice()
    # Select device add to torch
    device = select_device(device=device)
    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    serve()
