from kivy.app import App
# from skimage.io import imsave
from kivy.clock import Clock
# from kivy.uix.widget import Widget
# import time
from kivy.config import Config
from kivy.uix.boxlayout import BoxLayout
# import pandas as pd
# import numpy as np
from kivy.uix.recycleview import RecycleView

from utils.databasetools import DataBase
# import utils.debugtools
from utils.facetools import WatchDog
from utils.imagetools import InputImage

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


# class DataCard(RelativeLayout):


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
    def build(self):
        self.cm = CameraClick()
        self.db = DataBase(DATABASE_ADDRESS)
        Clock.schedule_interval(self.update, 1.0 / 33.0)

        self.dog = WatchDog(detector_path=DETECTOR_PATH,
                            predictor_path=PREDICTOR_PATH,
                            recognizer_path=RECOGNIZER_PATH)

        self.input_image = InputImage('images/facelogo010-01.png')
        self.input_image.change_source(0)

        return self.cm

    # def show_people_info(self, result_list):
    #     #show info of new people
    #         # if a person is in the view , reset his removal timer

    #     # remove info of old people

    def update(self, dt):
        # pass
        if self.cm.ids['play'].state is 'down':
            # 1.capture a frame
            ret, frame = self.input_image.get_frame()

            if ret is False:
                self.cm.ids['play'].state = 'normal'
                # TODO: show a warning!
                # self.cm.ids['camera'].source = './images/facelogo09-01.png'
                self.cm.ids['camera'].reload()
            else:
                result_list = self.dog.identify(frame)
                if result_list is not None:
                    correspondence_dict = self.db.update(result_list)
                    self.db.save_data_frame()
                    for i in range(len(result_list['DetectedFaces'])):
                    ###face boxes#################################################
                        rect = result_list['DetectedFaces'][i].rect
                        if self.cm.ids['face_box'].active:
                            frame = self.input_image.add_face_boxes(frame, rect)
                    ###show names#################################################
                        if self.cm.ids['names'].active:
                            id_num = correspondence_dict[i]
                            text = self.db.dataframe.at[
                                id_num, 'FirstName'] + ' ' + self.db.dataframe.at[
                                    id_num, 'LastName']

                            frame = self.input_image.add_name(frame, text, rect)
                    ##############################################################

                # self.input_image.update_cv2_window(frame)
                texture = self.input_image.get_kivy_texture(frame)
                self.cm.ids['camera'].color = (1, 1, 1, 1)
                self.cm.ids['camera'].texture = texture


if __name__ == '__main__':
    ZharfaApp().run()
