import time
from detect import detect
import cv2
import numpy as np
from copy import deepcopy
from utils.face_processing import get_coordinates_with_margin
from datetime import datetime, timedelta
from utils import config
from utils.torch_utils import select_device
import pygame
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

pygame.mixer.init()
def alert(audio, timer, alert_time):
    if datetime.now().time() > alert_time[0].time():
        interval = timedelta(seconds=timer)
        pygame.mixer.Sound(audio).play()
        time.sleep(0.02)
        pygame.mixer.Sound(audio).stop()
        time.sleep(0.02)
        alert_time.clear()
        alert_time.append(datetime.now()+interval)
    else:
        pass

def send_email(img, trackid, s, msg):

    _, buffer = cv2.imencode(".jpg", img)
    image_bytes = (buffer.tobytes())
    p=MIMEBase('application','octet-stream')
    p.set_payload(image_bytes)

    encoders.encode_base64(p)

    p.add_header('Content-Disposition',"attachment; filename=%s"%str(trackid))
    msg.attach(p)

    text=msg.as_string()

    s.sendmail("detdetxauxi9x@gmail.com","tittet98@gmail.com",text)

def leave_object_detect(model, frame_generator, image_size, iou_thres, conf_thres, device, sort_tracker):
    for frame, _ in frame_generator:
        name_class, counted_vehicle, timestamp, bounding_boxes, scores, response_bounding_boxes = detect(model, frame, image_size, iou_thres, conf_thres, device)
        time_arrival = time.time()
        draw_frame = deepcopy(frame)
        obj_detect = []
        if len(scores) > 0:
            response_bounding_boxes = np.array(
                [get_coordinates_with_margin(draw_frame, box) for box in response_bounding_boxes]).reshape(-1, 4)
            # Change confidence to (N, 1)
            scores = np.array(scores)
            detections = response_bounding_boxes.astype(np.float32)

            # Do tracking
            trackers = sort_tracker.update(detections)

            for i, tracker in enumerate(trackers):
                trackers[i].bbox = response_bounding_boxes[i]
                trackers[i].confidence = scores[i]
                trackers[i].name_class = name_class[i]
                time_alive = time_arrival - tracker.time_track
                if time_alive > 2:
                    dist = distance_object(response_bounding_boxes[i], tracker.box)
                    if dist > 100:
                        cv2.rectangle(frame,
                                      (int(tracker.box[0]), int(tracker.box[1]))
                                      , (int(tracker.box[2]), int(tracker.box[3])),
                                      [0, 255, 0], 3)
                        cv2.rectangle(frame,
                                      (int(response_bounding_boxes[i][0]), int(response_bounding_boxes[i][1]))
                                      , (int(response_bounding_boxes[i][2]), int(response_bounding_boxes[i][3])),
                                      [0, 0, 233], 3)
                        cv2.putText(frame, str(tracker.time_track),
                                    (int(response_bounding_boxes[i][0]), int(response_bounding_boxes[i][1])), 0, 1,
                                    [225, 0, 0], thickness=4,
                                    lineType=cv2.LINE_AA)
                        cv2.putText(frame, str(dist),
                                    (int(response_bounding_boxes[i][0]), int(response_bounding_boxes[i][1])-100), 0, 1,
                                    [0, 0, 255], thickness=4,
                                    lineType=cv2.LINE_AA)
                        obj_detect.append(trackers[i])
        yield timestamp, obj_detect, frame


def violate_object_detect(model, frame_generator, sort_tracker, areas, alert_time, list_track):
    alert_time=[datetime.now()]
    image_size = int(config.get_image_size())
    iou_thres = float(config.get_iou_threshold())
    conf_thres = float(config.get_confidence_threshold())
    # get device from file config
    device = config.get_divice()
    # Select device add to torch
    device = select_device(device=device)
    for frame, _timestamp in frame_generator:
        name_class, counted_vehicle, timestamp, bounding_boxes, scores, response_bounding_boxes = detect(model, frame, image_size, iou_thres, conf_thres, device)
        draw_frame = deepcopy(frame)
        is_send = False
        for area in areas:
            if area.poly:
                contour = [(int(point.x), int(point.y)) for point in area.poly]
                cv2.drawContours(frame, [np.array(contour)], 0, [255,255,0], 2)
        if len(scores) > 0:
            response_bounding_boxes = np.array(
                [get_coordinates_with_margin(draw_frame, box) for box in response_bounding_boxes]).reshape(-1, 4)
            # Change confidence to (N, 1)
            scores = np.array(scores)
            detections = response_bounding_boxes.astype(np.float32)
                  # Do tracking
            trackers = sort_tracker.update(detections)
            # remove_dead_track_id(trackers, list_track)
            list_obj = []
            for i, tracker in enumerate(trackers):
                trackers[i].bbox = response_bounding_boxes[i]
                trackers[i].confidence = scores[i]
                trackers[i].name_class = name_class[i]
                # print(trackers[i].track_id)
                track_id = trackers[i].track_id
                if name_class[i] == 'person':
                    isInside = object_location(response_bounding_boxes[i], areas,frame)
                    print("Inside", isInside)
                    if isInside == True:
                        list_obj.append(trackers[i])
                        cv2.rectangle(frame,
                                      (int(response_bounding_boxes[i][0]), int(response_bounding_boxes[i][1]))
                                      , (int(response_bounding_boxes[i][2]), int(response_bounding_boxes[i][3])),
                                      [0, 0, 255], 3)
                        cv2.putText(frame, str(track_id), (int(response_bounding_boxes[i][0]), int(response_bounding_boxes[i][1])-10), cv2.FONT_HERSHEY_SIMPLEX,
                                            1, (255, 0, 0), 2, cv2.LINE_AA)
                        alert("/home/datdo1/Downloads/mixkit-toy-telephone-ring-1351.wav", 3, alert_time)

                        if not tracker.is_recognized:
                            print("Send:", track_id)
                            print("List_track:", list_track)
                            # send_email(frame, track_id, s, msg)
                            tracker.is_recognized = True
                            is_send = True
                            # cv2.imwrite(track_id +".jpg", frame)
                # list_track.append(tracker)
            if list_obj is not None:
                yield _timestamp, list_obj, frame, is_send
            # list_track = trackers
        # cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        # cv2.imshow("Image", frame)
        # cv2.waitKey(1)
def check_bbox_exis(bbox, list_obj_bbox):
    min_distance = None
    _obj = None
    #Find obj when number of obj has more 2 obj
    for obj in list_obj_bbox:
        dist = np.sqrt((obj.bbox[0] - bbox[0]) ** 2 + (obj.bbox[1] - bbox[1]) ** 2)
        if min_distance == None:
            _obj = obj
        else:
            if dist < min_distance:
                _obj = obj
        min_distance = dist

    #Check obj is move?
    if min_distance == None:
        return None
    else:
        if min_distance < 20:
            return _obj
        return None

def bboxs_around(bbox, list_bbox):
    min_distance = 200
    _obj = None
    for obj in list_bbox:
        dist = np.sqrt((obj[0] - bbox[0]) ** 2 + (obj[1] - bbox[1]) ** 2)

        if dist < min_distance and dist != 0:
            _obj = obj
            return _obj
    return None

def del_list_bbox(list_bbox):
    time_now = time.time()
    for bb in list_bbox:
        time_alive = time_now - bb.time_arrival
        if time_alive > 100000:
            list_bbox.remove(bb)

def object_location(bounding_box, areas,frame):
    center_x = int((bounding_box[0] + bounding_box[2]) / 2 * 10000)
    center_y = int((bounding_box[1] + bounding_box[3]) / 2 * 10000)
    for area in areas:
        if area.poly:
            contour = [(int(point.x * 10000), int(point.y * 10000)) for point in area.poly]
            point = cv2.pointPolygonTest(np.array(contour), (center_x, center_y), False)
            print("Point:", point)
            if point > 0:
                return True
    return False
def distance_object(bounding_box_1, bounding_box_2):

    center_x_1 = int((bounding_box_1[0] + bounding_box_1[2]) / 2)
    center_y_1 = int((bounding_box_1[1] + bounding_box_1[3]) / 2)
    center_x_2 = int((bounding_box_2[0] + bounding_box_2[2]) / 2)
    center_y_2 = int((bounding_box_2[1] + bounding_box_2[3]) / 2)
    dist = np.sqrt((center_x_1 - center_x_2) ** 2 + (
                center_y_1 - center_y_2) ** 2)

    return dist



def remove_dead_track_id(trackers, track_list):
    """
    Remove dead track_id from self.track_list
    Args:
        trackers: list of KalmanBoxTracker returned from Sort.update()
    """
    for tracker in track_list:  # Avoid problem when pop using track_id
        if tracker.track_id not in [tracker.track_id for tracker in trackers]:
            track_list.pop(tracker)

# def update_recognition(tracker):
#     """
#     Update recognition information: confidence, face_token, ...
#     Args:
#         tracker: a KalmanBoxTracker
#     """
#     if tracker.track_id in self.track_list.keys() and \
#             tracker.labeled_face.confidence <= self.track_list[tracker.track_id].confidence:
#         # Mark recognized
#         if not tracker.is_recognized:
#             tracker.is_recognized = True
#
#         # Update result of current tracker
#         tracker.labeled_face = self.track_list[tracker.track_id]
