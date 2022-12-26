import numpy as np
import torch
from numpy import random

from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
from collections import defaultdict

def detect(model, im0s, image_size, iou_thres, conf_thres, device):
    # Initialize
    set_logging()

    half = device.type != 'cpu'  # half precision only supported on CUDA

    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(image_size, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    # process input
    img = letterbox(im0s, new_shape=image_size, stride=32)[0]
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    t1 = time_synchronized()
    pred = model(img)[0]

    # Apply NMS
    pred = non_max_suppression(pred, conf_thres=conf_thres, iou_thres=iou_thres, classes=None, agnostic=False)
    t2 = time_synchronized()
    name_vehicles = []
    counted_vehicles = []
    bounding_boxes = defaultdict(list)
    response_bounding_boxes = []
    confidences = []
    labels = []
    for i, det in enumerate(pred):  # detections per image
        s = ''
        s += '%gx%g ' % img.shape[2:]  # print string
        if len(det):
            # Rescale boxes from img_size to im0 size
            # get coors from detections
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                name_vehicles.append(names[int(c)])
                counted_vehicles.append(n)

            for *xyxy, conf, cls in reversed(det):

                label = f'{names[int(cls)]} {conf:.2f}'
                # plot_one_box(xyxy, im0s, label=label, color=colors[int(cls)], line_thickness=3)
                bounding_boxes[names[int(cls)]].append([int(xyxy[0]), int(xyxy[1]),
                                   int(xyxy[2]) - int(xyxy[0]), int(xyxy[3]) - int(xyxy[1])])
                response_bounding_boxes.append([int(xyxy[0]), int(xyxy[1]),
                                   int(xyxy[2]), int(xyxy[3])])

                confidences.append(conf)
                labels.append(names[int(cls)])




    print(f'{s}Done. ({t2 - t1:.3f}s)')


    return labels, counted_vehicles, t2, bounding_boxes, confidences, response_bounding_boxes