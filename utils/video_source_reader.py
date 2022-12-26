import itertools
import re
import cv2
import time

from SDK.sdk_iva import Ws2Decoder
from utils import error_code
from utils.application_properties import get_config_variable
from utils.customized_exception import BadRequestException
import logging

logger = logging.getLogger(__name__)


class VideoSourceReader:
    def __init__(self, source_url):
        self.web_socket_source_regex = "^ws://.*\|\|\d{1,4}x\d{1,4}$"
        self.decoder = None
        logger.info("Read from source_url {}".format(source_url))
        if source_url.startswith("ws:"):
            self.frame_generator = self.read_frame_from_websocket_source(source_url)
        else:
            self.frame_generator = self.read_from_rtsp_source(source_url)
        self.video_capture = None

    def read_from_rtsp_source(self, source_url):
        sending_generator = itertools.cycle(range(get_config_variable("skip_frame_count")))
        self.video_capture = cv2.VideoCapture((source_url))
        while True:
            success, frame = self.video_capture.read()
            if not success:
                raise BadRequestException(error_code.CANNOT_READ_SOURCE_URL, "source-url", source_url)
            sending_index = next(sending_generator)
            if sending_index % get_config_variable("skip_frame_count") != 0:
                continue
            yield frame, time.time()

    def read_frame_from_websocket_source(self, source_url):
        sending_generator = itertools.cycle(range(get_config_variable("skip_frame_count")))
        # if not re.match(self.web_socket_source_regex, source_url):
        #     raise BadRequestException(error_code.INVALID_WEBSOCKET_SOURCE_URL, "source_url", source_url)
        # source_url = source_url.rsplit("||")
        decoder = Ws2Decoder(url=source_url,outputFormat="yuv420p",debugMode=False)
        decoder.start()
        self.decoder = decoder
        while True:
            ret, frame, timestamp = self.decoder.recv()
            print(timestamp)

            if frame is None:
                time.sleep(0.1)
            if not ret:
                self.decoder.stop()
                raise BadRequestException(error_code.CANNOT_READ_SOURCE_URL, "source_url", source_url)
            sending_index = next(sending_generator)
            if sending_index % get_config_variable("skip_frame_count") != 0:
                continue
            if timestamp:
                yield cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_I420), timestamp

    def remove_source(self):
        if self.decoder is not None:
            self.decoder.stop()

        if get_config_variable("debug_mode"):
            cv2.destroyAllWindows()
