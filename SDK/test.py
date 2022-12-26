# from sdk_v1 import Ws2Decoder
from sdk_iva import Ws2Decoder
import time
import  cv2
import threading
import sys

def stop(obj):
    time.sleep(10)
    print('stop#####################################')
    obj.stop()

# url = 'ws://10.60.110.163/evup/12541_161035210/110000000099xyz96719'
# url = "ws://10.60.110.163/evup/12541_1613992235/110000000006xyz96709"
# url = "ws://nvr.thinghub.vn/evup/12541_1613992235/110000000006xyz96709"
#url = "ws://10.61.185.108:8005/evup/12541_161399223253322/110000000006xyz96709 "
url = "ws://nvr.thinghub.vn/evup/12541_161e123422xxz/110000000006xyz96709"

# url = "ws://10.60.156.195/evup/12548_16145675189/330000000027xyz96805"
width = 640
height = 480
test = Ws2Decoder(url, True)
test.start()
# stopThread = threading.Thread(target=stop, args=(test, ))
# stopThread.start()
try:
    while True:
        ret, frame, timestamp = test.recv()
        if ret:
            if timestamp == None:
                time.sleep(0.01)
                #pass
                # print('waiting......')
            else:
                # cv2.imwrite("./debug_images/{}.jpg".format(timestamp), frame)
                # cv2.imshow("image", frame)
                # if cv2.waitKey(1) == ord('q'):
                #     break
                print('iva timeFrame = {}'.format(timestamp))
                time.sleep(0.01)
        else:
            print('stop from iva')
            test.stop()
            break
    # stopThread.join()
    test.join()
except KeyboardInterrupt:
    # stopThread.join()
    print("keyboard interupt")
    test.stop()
    test.join()
    sys.exit()
except Exception as e:
    # stopThread.join()
    print(e)
    print('something is wrong!')
    test.stop()
    test.join()
    sys.exit()