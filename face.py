from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.widget import Widget
import time
from kivy.config import Config
# from skimage.io import imread
from kivy.clock import Clock
from kivy.graphics.texture import Texture
import cv2
import dlib
import pandas as pd
import numpy as np


# from kivy.core.window import Window

# Window.borderless = True

Config.set('kivy','window_icon','./images/facelogo010-01.png')
# Config.set('graphics','borderless',1)

face_detector_path = "./utils/core/mmod_human_face_detector.dat"
predictor_path = "./utils/core/shape_predictor_5_face_landmarks.dat"
face_rec_model_path = "./utils/core/dlib_face_recognition_resnet_model_v1.dat"
database_address = './database/database_fixed_compress.hdf'

dbase = pd.read_hdf(database_address)

def compare(vec1, vec2):
    return np.linalg.norm(vec1 - vec2) < 0.6

class CameraClick(BoxLayout):
    def capture(self):
        '''
        Function to capture the images and give them the names
        according to their captured time and date.
        '''
        camera = self.ids['camera']
        # timestr = time.strftime("%Y%m%d_%H%M%S")
        # camera.export_to_png("IMG_{}.png".format(timestr))
        print("Captured")



class ZharfaApp(App):

    def build(self):
        # print( app.get_application_icon())
        # self.detector = dlib.get_frontal_face_detector()
        self.face_detector = dlib.cnn_face_detection_model_v1(face_detector_path)
        self.sp = dlib.shape_predictor(predictor_path)
        self.facerec = dlib.face_recognition_model_v1(face_rec_model_path)
        self.cm = CameraClick()
        self.capture = cv2.VideoCapture(1)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH,1920/2)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT,1080/2)
        # cv2.namedWindow("CV2 Image")
        Clock.schedule_interval(self.update, 1.0/33.0)
        return self.cm

    def update(self, dt):

        if(self.cm.ids['play'].state is 'down'):
            # 1.capture a frame
            ret, frame = self.capture.read()
            # 2.detecting faces
            # dets = self.detector(frame, 1)
            dets = self.face_detector(frame)

            for i, subject in enumerate(dets): 
                subject = subject.rect               
                cv2.rectangle(frame,(subject.right(),subject.top()),(subject.left(),subject.bottom()),(0,255,0),3)
            
                shape = self.sp(frame, subject)
                face_chip = dlib.get_face_chip(frame, shape,150,0.25)
                face_descriptor_from_prealigned_image = self.facerec.compute_face_descriptor(face_chip)  

                text = "unknown"
                for index in dbase.index:
                    if compare(dbase.RecognitionID[index], np.array(face_descriptor_from_prealigned_image)):
                        text = dbase.FirstName[index] + ' ' + dbase.LastName[index]

                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame,text,(subject.left(),subject.bottom()+10), font, 1,(255,255,255),2,cv2.LINE_AA)

            # cv2.imshow("CV2 Image", frame)
            # convert it to texture
            buf1 = cv2.flip(frame, 0)
            buf = buf1.tostring()
            texture1 = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture1.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            # display image from the texture
            self.cm.ids['camera'].color = (1,1,1,1)
            self.cm.ids['camera'].texture = texture1

if __name__ == '__main__':
    ZharfaApp().run()
