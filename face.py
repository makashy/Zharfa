import os
import queue
import time
from multiprocessing import JoinableQueue, Process

from kivy.app import App
from kivy.clock import Clock
from kivy.config import Config
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.recycleview import RecycleView

from utils.databasetools import DataBase
# import utils.debugtools
from utils.facetools import DetectionProcess, IdentificationProcess
from utils.imagetools import InputImageProcess
from utils import gui

Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
Config.set('kivy', 'window_icon', './images/facelogo010-01.png')
Config.set('graphics', 'minimum_width', 425)
Config.set('graphics', 'minimum_height', 240)

DETECTOR_PATH = "./utils/core/mmod_human_face_detector.dat"
PREDICTOR_PATH = "./utils/core/shape_predictor_5_face_landmarks.dat"
# recognizer_path = "./utils/core/dlib_face_recognition_resnet_model_v1.dat"
RECOGNIZER_PATH = './utils/core/ms_celeb_1M_facenet_keras_weights.h5'
DATABASE_ADDRESS = './database/'


class RV(RecycleView):
    def __init__(self, **kwargs):
        super(RV, self).__init__(**kwargs)
        self.data = [{'text': str(x)} for x in range(100)]


class CameraClick(BoxLayout):
    def capture(self):
        '''
        Function to capture the images and give them the names
        according to their captured time and date.
        '''
        # camera = self.ids['camera']
        # timestr = time.strftime("%Y%m%d_%H%M%S")
        # camera.export_to_png("IMG_{}.png".format(timestr))
        print("Captured")


class ZharfaApp(App):
    def __init__(self):
        super().__init__()
        self.time = time.time()
        self.counter = 0
        self.cm = None
        self.database = None

        self.input_process = InputImageProcess('images/Demo1.mkv',
                                               'images/Demo1.mkv',
                                               'images/Demo2.mp4')

        self.detection_process = DetectionProcess(DETECTOR_PATH, 0, self.input_process.image_list_q)
        self.identification_process = IdentificationProcess(PREDICTOR_PATH, RECOGNIZER_PATH, 0, self.detection_process.output_q)

        self.timer = time.time()
        self.flag = 0

    def build(self):
        self.cm = CameraClick()
        self.database = DataBase(DATABASE_ADDRESS)
        Clock.schedule_interval(self.update, 1.0 / 33.0)

        self.input_process.start()
        self.detection_process.start()
        self.identification_process.start()

        return self.cm

    def update_save_database(self, result_list):
        correspondence_dict = None
        if result_list is not None:
            correspondence_dict = self.database.update(result_list)
            self.database.save_data_frame()
            return correspondence_dict

    def update_frame_viewer(self, result_list, frame, correspondence_dict):
        if result_list is not None:
            for i in range(result_list['Size']):
                rect = result_list['DetectionRects'][i]
                ###face boxes#################################################
                if self.cm.ids['face_box'].active:
                    frame = gui.add_face_boxes(frame, rect)
                ###show names#################################################
                if self.cm.ids['names'].active:
                    id_num = correspondence_dict[i]
                    text = self.database.dataframe.at[
                        id_num, 'FirstName'] + ' ' + self.database.dataframe.at[
                            id_num, 'LastName']
                    frame = gui.add_name(frame, text, rect)
                ###show ids#################################################
                if self.cm.ids['ids'].active:
                    frame = gui.add_id(frame, str(correspondence_dict[i]),
                                       rect)
                ##############################################################
        else:
            print("No detection at :", time.time())

        print("OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO", time.time())
        texture = gui.get_kivy_texture(frame)
        self.cm.ids['camera'].color = (1, 1, 1, 1)
        self.cm.ids['camera'].texture = texture

    def update(self, dt):
        # TODO: Improve this ↓!
        if time.time() - self.time > 1:
            self.time = time.time()
            self.cm.ids['FPS'].text = str(self.counter)
            self.counter = 0

        if self.cm.ids['play'].state is 'down':
            usable, item = self.identification_process.output_q.empty_and_get()
            if usable:
                print("66666666666666666666 update starts at: ", time.time())
                if item is None:
                    self.cm.ids['play'].state = 'normal'
                    self.cm.ids['camera'].reload()
                    self.input_process.play_mode_q.empty_and_put(False)
                    # TODO: show a warning!
                else:
                    frame, result_list = item
                    self.counter = self.counter + 1
                    correspondence_dict = self.update_save_database(result_list)
                    self.update_frame_viewer(result_list, frame, correspondence_dict)
                    print("77777777777777777 update finishes at: ", time.time())

    def on_stop(self):

        self.input_process.end_process() 
        self.detection_process.end_process()
        self.identification_process.end_process()


if __name__ == '__main__':
    ZharfaApp().run()
