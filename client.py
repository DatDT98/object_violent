import sys

import grpc
from protos import object_server_pb2_grpc as service_grpc
from protos import object_server_pb2 as service
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import datetime
import cv2
import os
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'

def send_email(img, timestamp):

    # _, buffer = cv2.imencode(".jpg", img)
    # image_bytes = (buffer.tobytes())
    p=MIMEBase('application','octet-stream')
    p.set_payload(img)

    encoders.encode_base64(p)

    p.add_header('Content-Disposition',"attachment; filename=%s"%str(timestamp))
    msg.attach(p)

    text=msg.as_string()

    s.sendmail("detdetxauxi9x@gmail.com","tittet98@gmail.com",text)
fromaddr = "detdetxauxi9x@gmail.com"  # sender gmail address
toaddr = "tittet98@gmail.com"  # reciver gmail address
msg = MIMEMultipart()
msg['From'] = fromaddr
msg['To'] = toaddr
msg['Subject'] = "found"
body = "sent from datdat98"
msg.attach(MIMEText(body, 'plain'))
s = smtplib.SMTP('smtp.gmail.com', 587)
s.starttls()
s.login(fromaddr, "Abcd1234.")  # enter sender gmail password here
# make an RPC request
# channel = grpc.insecure_channel('10.60.108.45:9080')
path = "rtsp://admin:Datdat98@192.168.1.160/onvif1"
channel = grpc.insecure_channel('localhost:50051')

stub = service_grpc.ObjectServiceStub(channel)
metadata = [('api_key', '5GpbfutWaHoJllrhsDtE9UJVDpOwlBFW')]

# request = service.ObjectRequest(ws="rtsp://root:1111@117.6.121.13:554/axis-media/media.amp")
# response = stub.ForgotObjectDetect(service.ObjectRequest(ws="rtsp://root:1111@117.6.121.13:554/axis-media/media.amp"), metadata=metadata)
# response = stub.CountObjectVideo(service.ObjectRequest(ws="/home/datdt/Downloads/video/Object_sign.mp4"))
# box = service.Box(x=300,y=200,width=450,height=500)
poly = []

video = cv2.VideoCapture((path))
ok, frame = video.read()

# Define an initial bounding box
# bbox = (287, 23, 86, 320)

# Uncomment the line below to select a different bounding box
bbox = cv2.selectROI(frame, False)

video.release()
cv2.destroyAllWindows()
# cv2.waitKey(0)
p1 = service.Point(x=int(bbox[0]), y=int(bbox[1]))
# p2 = service.Point(x=1000, y = 1000)
p2 = service.Point(x=int(bbox[0] + bbox[2]), y = int(bbox[1]))
p3 = service.Point(x=int(bbox[0] + bbox[2]), y=int(bbox[1] + bbox[3]))
p4 = service.Point(x=int(bbox[0]), y = int(bbox[1] + bbox[3]))

poly.append(p1)
poly.append(p2)
poly.append(p3)
poly.append(p4)

areas = service.Area(area_id='1', poly=poly)
response = stub.ViolateObjectDetect(service.ObjectRequest(ws=path, areas=[areas]), metadata=metadata)
# request = service.StreamingRequest(source_url="rtsp://admin:abcd1234@172.16.10.84:554/Channels/101")
# results = stub.DetectFire(request)
for result in response:
    # for i, _ in enumerate(result.name):
    #     print(result.name[i], " ", result.count[i])
    # print("Timestamp: ", result.forgot_obj)
    if result.send == True:
        print(result.send)
        send_email(result.image_bytes, datetime.datetime.fromtimestamp(result.timestamp))
    # for Object in result.counted_Objects:
    #     for detail in Object.detail:
    #         print("Object: ", detail.Object_type , " has ", detail.count)



# import requests
# import json
#
# def sendSMS(msg,numbers):
#     headers = {
#     "authkey": "place AUTH-KEY here",
#     "Content-Type": "application/json"
#     }
#
#     data = "{ \"sender\": \"GTURES\", \"route\": \"4\", \"country\": \"91\", \"sms\": [ { \"message\": \""+msg+"\", \"to\": "+json.dumps(numbers)+" } ] }"
#
#     requests.post("https://api.msg91.com/api/v2/sendsms", headers=headers, data=data)
# sendSMS("hi, i'm detdet",[+84966250499])

# import smtplib
# from email.mime.multipart import MIMEMultipart
# from email.mime.text import MIMEText
# from email.mime.base import MIMEBase
# from email import encoders
#
#
# fromaddr="detdetxauxi9x@gmail.com"  #sender gmail address
# toaddr="tittet98@gmail.com"   #reciver gmail address
# msg=MIMEMultipart()
# msg['From']=fromaddr
# msg['To']=toaddr
# msg['Subject']="found"
# body="sent from datdat98"
# msg.attach(MIMEText(body,'plain'))
# filename="test.jpg"
# attachment=open("/home/datdo1/Downloads/bg.jpeg","rb") #image folder
#
# p=MIMEBase('application','octet-stream')
# p.set_payload((attachment).read())
#
# encoders.encode_base64(p)
#
# p.add_header('Content-Disposition',"attachment; filename=%s"%filename)
# msg.attach(p)
#
# s=smtplib.SMTP('smtp.gmail.com',587)
#
# s.starttls()
#
# s.login(fromaddr,"Abcd1234.") #enter sender gmail password here
#
# text=msg.as_string()
#
# s.sendmail(fromaddr,toaddr,text)
#
# s.quit()
# import base64
#
#
# img_path_selfie = "/home/datdo1/Downloads/bg.jpeg"
# #img_path_selfie = "glass.jpeg"
# image1 = open(img_path_selfie, 'rb')
# image_read1 = image1.read()
# img_selfie = base64.b64encode(image_read1)
# f = open("encode_img.png", "wb")
# f.write(img_selfie)
# f.close()
