import time

from kivy.graphics.texture import Texture #pylint: disable=no-name-in-module

import cv2 #pylint: disable=import-error


def open_cam_rtsp(uri, width, height, latency):
    gst_str = ('rtspsrc location={} latency={} ! '\
               'rtph264depay ! h264parse ! omxh264dec ! '\
               'nvvidconv ! '\
              'video/x-raw, width=(int){}, height=(int){}' \
               'format=(string)BGRx ! '\
               'videoconvert ! appsink').format(uri, latency, width, height)
    return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)


class InputImage():
    def __init__(self, initial_source, demo1_address, demo2_address):
        self.source = cv2.VideoCapture(initial_source)
        self.demo1_address = demo1_address
        self.demo2_address = demo2_address
        self.width = 1920
        self.height = 1080

    def change_source(self, new_source, mode, camera_number, width, hight):
        self.width = int(width)
        self.height = int(hight)
        self.source.release()
        if mode == 'Camera':
            self.source = cv2.VideoCapture(
                int(camera_number.replace('No.', '')) - 1)
        elif mode == 'IP Camera':
            if '192.168' in new_source:
                # self.source = open_cam_rtsp(new_source, 20000, self.width,
                #                             self.height)
                self.source = cv2.VideoCapture(new_source)
                print('IP camera ################')  #TODO remove
        elif mode == 'Video':
            self.source = cv2.VideoCapture(new_source)
        elif mode == 'Demo1':
            self.source = cv2.VideoCapture(self.demo1_address)
        elif mode == 'Demo2':
            self.source = cv2.VideoCapture(self.demo2_address)
        debug = self.source.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.source.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        print('#######################', debug)  #TODO remove

    def get_frame(self):
        for _ in range(5):
            ret, frame = self.source.read()
            if ret:
                frame = cv2.resize(frame, (self.width, self.height))
                break
            print('No data!')
            time.sleep(0.002)
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return ret, frame

    def do_with_timer(self, name='__str__', args=()):
        start_time = time.time_ns()
        result = self.__getattribute__(name)(*args)
        print(name + ": ", time.time_ns() - start_time)
        return result
