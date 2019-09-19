import time

import numpy as np
import cv2  #pylint: disable=import-error
from multiprocessing import JoinableQueue, Process
from utils.advanced_queues import AdvancedQueue


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
            time.sleep(0.002)
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return ret, frame

    def do_with_timer(self, name='__str__', args=()):
        start_time = time.time_ns()
        result = self.__getattribute__(name)(*args)
        print(name + ": ", time.time_ns() - start_time)
        return result


class InputImageProcess(Process):
    def __init__(self, initial_source, demo1_address, demo2_address):
        super().__init__()
        self.source_info_q = AdvancedQueue(1)
        self.image_list_q = AdvancedQueue(1)
        self.play_mode_q = AdvancedQueue(1)
        self.stop_signal_q = JoinableQueue(1)
        self.initial_source = initial_source
        self.demo1_address = demo1_address
        self.demo2_address = demo2_address

    def run(self):
        input_image = InputImage(self.initial_source, self.demo1_address, self.demo2_address)
        input_fps = min(40, input_image.source.get(cv2.CAP_PROP_FPS)) #TODO
        detection_fps = 2 #TODO
        frame_to_discard = np.int(np.ceil(input_fps / detection_fps)) #TODO
        discard_counter = 0 #TODO

        timer = time.time()
        state = False
        while True:
            ################################################################
            if self.stop_signal_q.empty() is False:
                input_image.source.release()
                break

            ################################################################
            usable, item = self.source_info_q.empty_and_get()
            if usable:
                info = item
                input_image.change_source(info[0], info[1], info[2], info[3],
                                          info[4])
                self.image_list_q.empty_out()

            ################################################################
            usable, item = self.play_mode_q.empty_and_get()
            if usable:
                state = item

            ################################################################
            if time.time() - timer > 1/input_fps and state:
                timer = time.time()
                frame_availability, frame = input_image.get_frame()

                #TODO: remove:
                if frame_availability is False:
                    print("No source")


                if discard_counter == frame_to_discard:
                    discard_counter = 0
                    self.image_list_q.empty_and_put(frame)#self.image_list_q.put_nowait(frame) TODO ?
                    print("11111111111111111111 Capture at: ", time.time())
                discard_counter = discard_counter + 1

    def end_process(self):
        self.stop_signal_q.put_nowait(True)
        time.sleep(1)  #TODO : change duration or remove!
        self.terminate()
        self.join()
        self.close()