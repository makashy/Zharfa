import time

import numpy as np
import cv2  #pylint: disable=import-error
from multiprocessing import JoinableQueue, Process


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


# def input_image_process(initial_source, demo1_address, demo2_address,
#                         source_info, image_list, play_mode, stop_signal):

#     input_image = InputImage(initial_source, demo1_address, demo2_address)
#     timer = time.time()
#     state = False
#     while True:
#         if stop_signal.empty() is False:
#             input_image.source.release()
#             break

#         if source_info.empty() is False:
#             info = source_info.get_nowait()
#             input_image.change_source(info[0], info[1], info[2], info[3],
#                                       info[4])

#             while image_list.empty() is False:
#                 image_list.get_nowait()

#         if play_mode.empty() is False:
#             state = play_mode.get_nowait()
#             print("OK")

#         if time.time() - timer > 0.5 and state:
#             timer = time.time()
#             # while (image_list.empty() is False):
#             #     image_list.get_nowait()

#             image_list.put_nowait(input_image.get_frame())


class InputImageProcess(Process):
    def __init__(self, initial_source, demo1_address, demo2_address):
        super().__init__()
        self.source_info = JoinableQueue()
        self.image_list = JoinableQueue()
        self.play_mode = JoinableQueue()
        self.stop_signal = JoinableQueue()
        self.initial_source = initial_source
        self.demo1_address = demo1_address
        self.demo2_address = demo2_address
        # self.proess = Process(target=input_image_process,
        #                       args=(initial_source, demo1_address,
        #                             demo2_address, self.source_info,
        #                             self.image_list, self.play_mode,
        #                             self.stop_signal))

    def run(self):
        input_image = InputImage(self.initial_source, self.demo1_address, self.demo2_address)
        input_fps = min(40, input_image.source.get(cv2.CAP_PROP_FPS)) #TODO
        detection_fps = 2 #TODO
        frame_to_discard = np.int(np.ceil(input_fps / detection_fps)) #TODO
        discard_counter = 0 #TODO

        timer = time.time()
        state = False
        while True:
            if self.stop_signal.empty() is False:
                input_image.source.release()
                break

            if self.source_info.empty() is False:
                info = self.source_info.get_nowait()
                input_image.change_source(info[0], info[1], info[2], info[3],
                                          info[4])

                while self.image_list.empty() is False:
                    self.image_list.get_nowait()

            if self.play_mode.empty() is False:
                state = self.play_mode.get_nowait()
                print("OK")

            if time.time() - timer > 1/input_fps and state:
                timer = time.time()
                # while (self.image_list.empty() is False):
                #     self.image_list.get_nowait()
                frame = input_image.get_frame()
                if discard_counter == frame_to_discard:
                    discard_counter = 0
                    self.image_list.put_nowait(frame)
                discard_counter = discard_counter + 1
                # self.image_list.put_nowait(input_image.get_frame())

    def end_process(self):
        self.stop_signal.put_nowait(True)
        time.sleep(1)  #TODO : change duration or remove!
        self.terminate()
        self.join()
        self.close()