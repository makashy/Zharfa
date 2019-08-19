import time

from kivy.graphics.texture import Texture

import cv2


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

    def check_source(self, mode, root_widget):
        if mode == 'Camera':
            root_widget.ids['source_address'].disabled = True
            root_widget.ids['camera_number'].disabled = False
        elif mode == 'IP Camera':
            root_widget.ids['source_address'].disabled = False
            root_widget.ids['camera_number'].disabled = True
        elif mode == 'Video':
            root_widget.ids['source_address'].disabled = False
            root_widget.ids['camera_number'].disabled = True
        elif 'Demo' in mode:
            root_widget.ids['source_address'].disabled = True
            root_widget.ids['camera_number'].disabled = True

    def change_source(self, new_source, mode, camera_number, width, hight):
        self.width = int(width)
        self.height = int(hight)
        self.source.release()
        if mode == 'Camera':
            self.source = cv2.VideoCapture(
                int(camera_number.replace('No.', '')) - 1)
        elif mode == 'IP Camera':
            if '192.168' in new_source:
                self.source = open_cam_rtsp(new_source, 20000, self.width,
                                            self.height)
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

    def add_face_boxes(self, frame, rect, color=(0, 255, 0), thickness=3):
        cv2.rectangle(frame, (rect.right(), rect.top()),
                      (rect.left(), rect.bottom()), color, thickness)
        return frame

    def add_name(self,
                 frame,
                 text,
                 rect,
                 font_face=cv2.FONT_HERSHEY_SIMPLEX,
                 font_scale=1,
                 color=(255, 255, 255),
                 thickness=2,
                 line_type=cv2.LINE_AA):
        cv2.putText(frame, text, (rect.left(), rect.bottom() + 10), font_face,
                    font_scale, color, thickness, line_type)  #TODO : 10?
        return frame

    def add_id(self,
               frame,
               text,
               rect,
               font_face=cv2.FONT_HERSHEY_SIMPLEX,
               font_scale=1,
               color=(255, 0, 0),
               thickness=2,
               line_type=cv2.LINE_AA):
        cv2.putText(frame, text, (rect.left(), rect.bottom() + 20), font_face,
                    font_scale, color, thickness, line_type)  #TODO : 20?
        return frame

    def get_kivy_texture(self, frame):
        buf = cv2.flip(frame, 0)
        buf = buf.tostring()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]),
                                 colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        return texture

    def open_cv2_window(self):
        cv2.namedWindow("Zharfa  (Large view)")

    def update_cv2_window(self, frame):
        cv2.imshow("Zharfa  (Large view)", frame)

    def close_cv2_window(self):
        cv2.destroyAllWindows()
        return result
