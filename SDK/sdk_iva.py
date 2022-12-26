import threading
import subprocess
import time
import re
import struct
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)

def date_now():
    return round(time.time() * 1000)


class Queue:
    def __init__(self, max_length=100):
        self.max_length = max_length
        self.queue_frame = []
        self.queue_time = []
        self.previous_get_time = 0
        self.previous_frame_time = 0

    def len(self):
        return len(self.queue_time)

    def enqueue(self, frame, timestamp):
        self.queue_time.append(timestamp)
        self.queue_frame.append(frame)
        if (len(self.queue_time) > self.max_length):
            self.queue_frame.pop(0)
            self.queue_time.pop(0)

    def dequeue(self):
        if len(self.queue_time) > 0:
            frame = self.queue_frame.pop(0)
            timestamp = self.queue_time.pop(0)
            if self.previous_get_time == 0:
                self.previous_get_time = date_now()
                adjust = 0
            else:
                now = date_now()
                adjust = now - self.previous_get_time
                self.previous_get_time = now
            while ((timestamp - self.previous_frame_time) < adjust) and (len(self.queue_time) > 0):
                frame = self.queue_frame.pop(0)
                timestamp = self.queue_time.pop(0)
            self.previous_frame_time = timestamp
            return frame, timestamp
        else:
            # time.sleep(0.03)
            return None, None

    def clear(self):
        self.queue_frame.clear()
        self.queue_time.clear()


class Ws2Decoder(threading.Thread):
    def __init__(self, url, outputFormat="rgb24", debugMode=False):
        threading.Thread.__init__(self)
        self.event = threading.Event()
        self.url = url
        self.channel_id = re.split("/|\?", url)[5]
        self.debug_mode = debugMode
        self.width = 0
        self.height = 0
        self.videoSize = 0
        self.fps = 0
        self.ret = True
        self.recv_time = date_now()
        self.queue = Queue()
        self.outputFormat = outputFormat

    def log_info(self, message):
        if (self.debug_mode == True):
            logger.info("({}) {}".format(self.channel_id, message))

    def log_error(self, message):
        logger.error("EXCEPTION: ({}) {}".format(self.channel_id, message))

    def init_sdk(self):
        try:
            dir_path = os.path.dirname(os.path.realpath(__file__))
            cmd = [dir_path + "/sdk-iva-decode-video"]
            cmd.append("-u")
            cmd.append(self.url)
            cmd.append("-t")
            cmd.append(self.outputFormat)
            if (self.debug_mode == True):
                cmd.append("-l")
            self.sdk_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, close_fds=True)
        except Exception as e:
            self.log_error("Init sdk error: {}".format(e))
            self.ret = False
            self.event.set()

    def run(self):
        self.log_info("Start sdk with ws_url {}".format(self.url))
        self.init_sdk()
        self.readOut = threading.Thread(target=self.readStdout, args=[])
        self.readOut.start()
        self.readErr = threading.Thread(target=self.readSterr, args=[])
        self.readErr.start()
        while not self.event.is_set():
            time.sleep(0.5)
        self.log_info("Stop sdk")
        self.queue.clear()
        self.readOut.join()
        self.readErr.join()

    def stop(self):
        self.event.set()
        self.sdk_process.kill()

    def recv(self):
        if self.width == 0 and self.height == 0:
            return self.ret, None, None
        # if (date_now() - self.recv_time) > 15000:
        #     self.log_error("Do not receive data within {}ms".format(date_now() - self.recv_time))
        #     return False, None, None
        frame, timestamp = self.queue.dequeue()
        if timestamp != None:
            self.log_info("length queue frame_time: {}".format(self.queue.len()))
            if (self.outputFormat == "rgb24"):
                reshape_frame = (
                    np
                        .frombuffer(frame, np.uint8)
                        .reshape(self.height, self.width, 3)
                )
            if (self.outputFormat == "yuv420p"):
                reshape_frame = (
                    np
                        .frombuffer(frame, np.uint8)
                        .reshape(self.height * 3 // 2, self.width)
                )
            return self.ret, reshape_frame, timestamp
        return self.ret, frame, timestamp

    def readStdout(self):
        while not self.event.is_set():
            if self.videoSize != 0:
                in_bytes = self.sdk_process.stdout.read(self.videoSize)
                if len(in_bytes) != (self.videoSize):
                    self.log_error("length package invaild")
                    self.ret = False
                    break

                if self.event.is_set():
                    self.ret = False
                    break
                self.log_info("stdout from python: {}".format(len(in_bytes)))
                timestamp = int(struct.unpack(">d", in_bytes[0:8])[0])
                self.log_info("frame timestamp: {}".format(timestamp))
                self.queue.enqueue(in_bytes[8:], timestamp)
            time.sleep(0.02)

    def readSterr(self):
        for line in iter(self.sdk_process.stderr.readline, ''):
            if self.event.is_set():
                break
            decoded_line = line.decode('UTF-8')
            if self.width == 0 and self.height == 0:
                result = re.search('fps=(\d+) pixFmt=(\d+) width=(\d+) height=(\d+)', decoded_line)
                if result != None:
                    self.fps = int(result.group(1))
                    self.width = int(result.group(3))
                    self.height = int(result.group(4))
                    if (self.outputFormat == "rgb24"):
                        self.videoSize = self.width * self.height * 3 + 8
                    if (self.outputFormat == "yuv420p"):
                        self.videoSize = self.width * self.height * 3 // 2 + 8
                    self.log_info("fps {} width {} height {}".format(self.fps, self.width, self.height))

            result = re.search('ERR:', decoded_line)
            if result != None:
                self.log_error("stderr from python: {}".format(decoded_line))
                self.ret = False
                break
            self.log_info("stderr from python: {}".format(decoded_line))
            time.sleep(0.01)






