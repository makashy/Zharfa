""" GUI module (in kivy)
"""
import time

import numpy as np
from kivy.app import App  # pylint: disable=import-error
from kivy.clock import Clock  # pylint: disable=import-error
from kivy.config import Config  # pylint: disable=import-error
from kivy.graphics.texture import Texture  # pylint: disable=no-name-in-module
from kivy.lang import Builder  # pylint: disable=import-error
from kivy.uix.boxlayout import BoxLayout  # pylint: disable=import-error

import cv2  # pylint: disable=import-error
import utils.tpi as tpi
from utils.chronometer import Chronometer

Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
Config.set('kivy', 'window_icon', "./images/facelogo010-01.png")
Config.set('graphics', 'minimum_width', 425)
Config.set('graphics', 'minimum_height', 240)
Builder.load_file('./utils/Zharfa.kv')


def check_source(mode, root_widget):  #TODO
    """ Manages GUI handles for changing input source
    """
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


def add_face_boxes(frame, rect, color=(0, 255, 0), thickness=3):
    """ adds boxes on a frame around detected faces
    """
    cv2.rectangle(frame, (rect.right(), rect.top()),
                  (rect.left(), rect.bottom()), color, thickness)
    return frame


def add_name(frame,
             text,
             rect,
             font_face=cv2.FONT_HERSHEY_SIMPLEX,
             font_scale=1,
             color=(255, 255, 255),
             thickness=2,
             line_type=cv2.LINE_AA):
    """ adds names on a frame under detected faces
    """
    cv2.putText(frame, text, (rect.left(), rect.bottom() + 10), font_face,
                font_scale, color, thickness, line_type)  #TODO : 10?
    return frame


def add_id(frame,
           text,
           rect,
           font_face=cv2.FONT_HERSHEY_SIMPLEX,
           font_scale=1,
           color=(255, 0, 0),
           thickness=2,
           line_type=cv2.LINE_AA):
    """ adds IDs on a frame under detected faces
    """
    cv2.putText(frame, text, (rect.left(), rect.bottom() + 20), font_face,
                font_scale, color, thickness, line_type)  #TODO : 20?
    return frame


def get_kivy_texture(frame):
    """ Converts a captured frame by cv2 to appropriate texture,
    to be displayed in GUI
    """
    buf = cv2.flip(frame, 0)
    buf = buf.tostring()
    texture = Texture.create(size=(frame.shape[1], frame.shape[0]),
                             colorfmt='bgr')
    texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
    return texture


####################################################
class MainWindow(BoxLayout):
    """ Root Widget in kivy GUI
    """


class ZharfaApp(App):
    """ kivy app class
    """
    def __init__(self, comm):
        super().__init__()
        self.comm = comm
        self.main_window = None
        self.time = time.time()
        self.counter = 0
        self.chronometer = Chronometer()

        self.frame = np.zeros([1080, 1920, 3], np.int8)
        self.frame_req = self.comm.Irecv(self.frame,
                                         source=tpi.INPUT_PROCESS,
                                         tag=tpi.FRAME)

    def build(self):
        self.main_window = MainWindow()
        Clock.schedule_interval(self.update, 1.0 / 100.0)
        return self.main_window

    def update(self, dt):

        if self.main_window.ids['play'].state is 'down':
            data = self.frame_req.test()
            if data[0]:
                self.main_window.ids['FPS'].text = str(
                    1 / self.chronometer.average_elapsed())
                self.chronometer.start()
                texture = get_kivy_texture(self.frame)
                self.main_window.ids['camera'].color = (1, 1, 1, 1)
                self.main_window.ids['camera'].texture = texture
                self.frame_req = self.comm.Irecv(self.frame,
                                                 source=tpi.INPUT_PROCESS,
                                                 tag=tpi.FRAME)

    def on_stop(self):
        """ Sends a stop message to other ptocesses
        """
        for i in range(1, self.comm.Get_size()):
            self.comm.send(True, dest=i, tag=tpi.STOP)

    def change_input_setting(self, setting):
        """ Sends the input source settings to INPUT_PROCESS
        """
        self.comm.send([setting, time.perf_counter()],
                       dest=tpi.INPUT_PROCESS,
                       tag=tpi.SETTING)

    def change_input_mode(self, mode):
        """ Sends the mode message to INPUT_PROCESS
        """
        self.comm.send([mode, time.perf_counter()],
                       dest=tpi.INPUT_PROCESS,
                       tag=tpi.MODE)
