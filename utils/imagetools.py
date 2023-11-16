""" Input Image Processing Module
"""
import time

import numpy as np

import cv2  # pylint: disable=import-error
import utils.tpi as tpi
from utils.chronometer import Chronometer

# def open_cam_rtsp(uri, width, height, latency):
#     gst_str = ('rtspsrc location={} latency={} ! '\
#                'rtph264depay ! h264parse ! omxh264dec ! '\
#                'nvvidconv ! '\
#                'video/x-raw, width=(int){}, height=(int){}' \
#                'format=(string)BGRx ! '\
#                'videoconvert ! appsink').format(uri, latency, width, height)
#     return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)


class InputImageProcess():
    def __init__(self, communicator, initial_source, demo1_address,
                 demo2_address):
        self.comm = communicator
        self.stop_buf = np.zeros(1)
        self.stop_req = self.comm.Irecv(self.stop_buf,
                                        source=tpi.GUI_PROCESS,
                                        tag=tpi.STOP)
        self.setting_req = self.comm.irecv(source=tpi.GUI_PROCESS,
                                           tag=tpi.SETTING)
        self.mode_req = self.comm.irecv(source=tpi.GUI_PROCESS, tag=tpi.MODE)

        self.play = False
        self.source = cv2.VideoCapture(initial_source)
        self.demo1_address = demo1_address
        self.demo2_address = demo2_address
        self.width = 1920
        self.height = 1080

        self.timer = 0
        self.input_fps = min(40, self.source.get(cv2.CAP_PROP_FPS))  #TODO
        detection_fps = 80  #TODO
        self.frames_to_discard = np.int(np.ceil(self.input_fps /
                                                detection_fps))  #TODO
        self.discard_counter = 0  #TODO

        self.chronometer = Chronometer()
        # self.frame = np.zeros([1080, 1920, 3], np.int8)
        # self.test = self.comm.Send_init(self.frame, dest=tpi.GUI_PROCESS, tag=tpi.FRAME)

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
        print('#######################', debug, flush=True)  #TODO remove
        print("*****************",
              self.source.get(cv2.CAP_PROP_FPS),
              flush=True)

    def get_frame(self):
        for _ in range(5):
            ret, frame = self.source.read()
            if ret:
                frame = cv2.resize(frame, (self.width, self.height))
                break
            time.sleep(0.002)
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return ret, frame

    def flush_frame(self):
        for _ in range(5):
            ret, frame = self.source.read()
            if ret:
                break
            time.sleep(0.002)
        return ret, frame

    def do_with_timer(self, name='__str__', args=()):
        start_time = time.time_ns()
        result = self.__getattribute__(name)(*args)
        print(name + ": ", time.time_ns() - start_time)
        return result

    def run(self):
        """ The method representing the process's activity.
        """
        while True:
            if self.stop_req.test()[0]:
                self.source.release()
                print("Input Image Process(#{}) Stopped".format(
                    self.comm.Get_rank()),
                      flush=True)
                break

            # Change source
            data = self.setting_req.test()
            if data[0]:
                info = data[1][0]
                self.change_source(info[0], info[1], info[2], info[3], info[4])
                self.setting_req = self.comm.irecv(source=tpi.GUI_PROCESS,
                                                   tag=tpi.SETTING)
                print("Transfer time(Change source):",
                      time.perf_counter() - data[1][1],
                      flush=True)

            # Play Mode
            data = self.mode_req.test()
            if data[0]:
                self.play = data[1][0]
                self.mode_req = self.comm.irecv(source=tpi.GUI_PROCESS,
                                                tag=tpi.MODE)
                print("Transfer time(Play Mode):",
                      time.perf_counter() - data[1][1],
                      flush=True)

            # Reading images
            if self.play:
                frame_availability, frame = self.get_frame()
                self.comm.Send(frame, dest=tpi.GUI_PROCESS, tag=tpi.FRAME
                               )  # dest=tpi.DETECTION_PROCESS, tag=tpi.FRAME)
            # if time.time() - self.timer > 1/self.input_fps and self.play:
            #     self.timer = time.time()
            #     if self.discard_counter == 0:
            #         frame_availability, frame = self.get_frame()
            #         self.comm.Isend(frame,  dest=tpi.GUI_PROCESS, tag=tpi.FRAME
            #                         )# dest=tpi.DETECTION_PROCESS, tag=tpi.FRAME)
            #     else:
            #         self.flush_frame()

            #     self.discard_counter = self.discard_counter + 1
            #     if self.discard_counter == self.frames_to_discard:
            #         self.discard_counter = 0
