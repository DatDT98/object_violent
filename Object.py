# import time
# import random
# import datetime
# import telepot
# import sys
# def handle(msg):
#     chat_id = msg['chat']['id']
#     command = msg['text']
#     bot.send_photo(chat_id, photo=open('921280447421408919.jpg', 'rb'))
#
# bot = telepot.Bot('5265745775:AAEs4opxWZ-jiTVQlLHV5Pz25uRAfj4WEKs')
# bot.message_loop(handle)
# print("done")
import cv2
import os

os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'
vcap = cv2.VideoCapture("rtsp://admin:Datdat98@192.168.1.159/onvif1")
# vcap.set(cv2.CAP_PROP_FPS, 5.0)
if not vcap.isOpened():
    print('Cannot open RTSP stream')
    exit(-1)
while(1):
    ret, frame = vcap.read()
    print("reading")
    # if frame is not None:
    cv2.imshow('VIDEO', frame)
    cv2.waitKey(1)