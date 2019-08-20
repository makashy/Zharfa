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


def f(face_detector_path, predictor_path, keras_weight, frame_queue,
      result_queue, num, device_id):

    dog = WatchDog(detector_path=face_detector_path,
                   predictor_path=predictor_path,
                   recognizer_path=keras_weight,
                   device_id=device_id)

    while True:
        try:
            frame = frame_queue.get_nowait()
            print(num, "Received frame at ", time.time())
            if frame is False:
                break
            else:
                start_time = time.time()
                result_list = dog.identify(frame)
                print(" ||||||| Process time of  ", num, "  :  ",
                      time.time() - start_time)
                print(" ======= End time of  ", num, "  :  ", time.time())
                result_queue.put([result_list, frame])
        except queue.Empty:
            # pass
            print("######## No frame for", num)


class ZharfaApp(App):
    def __init__(self):
        super().__init__()
        self.cm = None
        self.database = None
        self.dog = None
        self.input_image = None

        self.num_dog = 1

        self.frame_queues = [JoinableQueue() for _ in range(self.num_dog)]
        self.result_queues = [JoinableQueue() for _ in range(self.num_dog)]
        self.dogs = [
            Process(target=f,
                    args=(DETECTOR_PATH, PREDICTOR_PATH, RECOGNIZER_PATH,
                          self.frame_queues[i], self.result_queues[i], i, i))
            for i in range(self.num_dog)
        ]

        self.timer = time.time()
        self.flag = 0

    def build(self):
        self.cm = CameraClick()
        self.database = DataBase(DATABASE_ADDRESS)
        Clock.schedule_interval(self.update, 1.0 / 33.0)

        self.input_image = InputImage('images/facelogo010-01.png',
                                      'images/Demo1.mkv', 'images/Demo2.mp4')
        self.input_image.change_source('',
                                       mode='Camera',
                                       camera_number='No.1',
                                       width=1920,
                                       hight=1080)

        for i in range(self.num_dog):
            self.dogs[i].start()

        return self.cm

    def update_save_database(self, result_list):
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
                    frame = self.input_image.add_face_boxes(frame, rect)
                ###show names#################################################
                if self.cm.ids['names'].active:
                    id_num = correspondence_dict[i]
                    text = self.database.dataframe.at[
                        id_num, 'FirstName'] + ' ' + self.database.dataframe.at[
                            id_num, 'LastName']
                    frame = self.input_image.add_name(frame, text, rect)
                ###show ids#################################################
                if self.cm.ids['ids'].active:
                    frame = self.input_image.add_id(
                        frame, str(correspondence_dict[i]), rect)
                ##############################################################
        else:
            print("No detection at :", time.time())

        print("Screen update at :", time.time())
        texture = self.input_image.get_kivy_texture(frame)
        self.cm.ids['camera'].color = (1, 1, 1, 1)
        self.cm.ids['camera'].texture = texture

    def update(self, dt):
        # pass
        if self.cm.ids['play'].state is 'down':
            # 1.capture a frame
            ret, frame_original = self.input_image.get_frame()

            if ret is False:
                self.cm.ids['play'].state = 'normal'
                # TODO: show a warning!
                self.cm.ids['camera'].reload()

            else:
                if time.time() - self.timer > 0.200:
                    self.timer = time.time()
                    # try:
                    #     self.frame_queues[self.flag].put_nowait(frame)
                    # except queue.Full:
                    #     print(self.flag, " didn't get frame")
                    if self.frame_queues[self.flag].empty():
                        self.frame_queues[self.flag].put_nowait(frame_original)
                        print("sent to ", self.flag, " at ", time.time())
                    else:
                        pass
                        # print(self.flag, " didn't get frame")
                    self.flag = self.flag + 1
                    if self.flag == self.num_dog:
                        self.flag = 0

            for i in range(self.num_dog):
                try:
                    result_list, frame = self.result_queues[i].get_nowait()
                    print("received from ", i, " at ", time.time())
                    correspondence_dict = self.update_save_database(
                        result_list)
                    self.update_frame_viewer(result_list, frame,
                                             correspondence_dict)
                except queue.Empty:
                    print("Not received any things at ", time.time())

    def on_stop(self):
        for i in range(self.num_dog):
            self.frame_queues[i].put_nowait(False)
        time.sleep(1)
        print('after sleep')
        for dog in self.dogs:

            dog.terminate()
            dog.join()
            dog.close()


if __name__ == '__main__':
    ZharfaApp().run()
