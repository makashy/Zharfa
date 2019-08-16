from kivy.app import App
# from skimage.io import imsave
from kivy.clock import Clock
# from kivy.uix.widget import Widget
# import time
from kivy.config import Config
from kivy.graphics.texture import Texture
from kivy.uix.boxlayout import BoxLayout
# import pandas as pd
# import numpy as np
from kivy.uix.recycleview import RecycleView

import cv2
from utils.databasetools import DataBase
# import utils.debugtools
from utils.facetools import WatchDog

Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
Config.set('kivy', 'window_icon', './images/facelogo010-01.png')
Config.set('graphics', 'minimum_width', 425)
Config.set('graphics', 'minimum_height', 200)


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

        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920 / 2)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080 / 2)
        # cv2.namedWindow("CV2 Image")

        return self.cm

    # def show_people_info(self, result_list):
    #     #show info of new people
    #         # if a person is in the view , reset his removal timer

    #     # remove info of old people

    def update(self, dt):
        # pass
        if self.cm.ids['play'].state is 'down':
            # 1.capture a frame
            for _ in range(10):
                ret, frame = self.capture.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 2.detecting faces
            # dets = self.dog.detector(frame)
            result_list = self.dog.identify(frame)
            if result_list is not None:
                correspondence_dict = self.db.update(result_list)
                self.db.save_data_frame()
                for i in range(len(result_list['DetectedFaces'])):
                    rect = result_list['DetectedFaces'][i].rect
                    # emb = result_list['RecognitionID'][i]
                    cv2.rectangle(frame, (rect.right(), rect.top()),
                                  (rect.left(), rect.bottom()), (0, 255, 0), 3)
                    id_num = correspondence_dict[i]
                    text = self.db.dataframe.at[
                        id_num, 'FirstName'] + ' ' + self.db.dataframe.at[
                            id_num, 'LastName']
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(frame, text, (rect.left(), rect.bottom() + 10),
                                font, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # cv2.imshow("CV2 Image", frame)
            # convert it to texture
            buf1 = cv2.flip(frame, 0)
            buf = buf1.tostring()
            texture1 = Texture.create(size=(frame.shape[1], frame.shape[0]),
                                      colorfmt='bgr')
            texture1.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            # display image from the texture
            self.cm.ids['camera'].color = (1, 1, 1, 1)
            self.cm.ids['camera'].texture = texture1


if __name__ == '__main__':
    ZharfaApp().run()
